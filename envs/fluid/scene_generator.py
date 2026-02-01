"""
MuJoCo to SPH Scene Generator

该模块提供从 MuJoCo 模型自动生成 SPH scene.json (RigidBodies 部分) 的功能。
支持坐标系转换、几何映射推断和 SPH 标记识别，确保两个仿真系统初始化的一致性。
"""

import json
import os
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import sys

# 设置日志
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('[SceneGenerator] %(levelname)s: %(message)s'))
    logger.addHandler(handler)


@dataclass
class GeomInfo:
    """几何信息数据类"""
    geom_type: str
    mesh_name: str = ""
    size: List[float] = None
    scale: List[float] = None
    is_static: bool = False  # 标记是否为静态刚体
    
    def __post_init__(self):
        if self.size is None:
            self.size = []
        if self.scale is None:
            self.scale = [1, 1, 1]


class SceneGenerator:
    """MuJoCo to SPH 场景生成器"""
    
    def __init__(self, env, config: Dict = None, config_path: str = None, runtime_config: Dict = None):
        """
        初始化 SceneGenerator
        
        Args:
            env: OrcaGymLocalEnv 实例
            config: 配置字典（场景模板配置，直接传入）
            config_path: 场景模板配置文件路径（优先从路径加载）
            runtime_config: 运行时配置（必须包含 orcalink_bridge.shared_modules.spring_force）
        """
        self.env = env
        self.config = self._load_config(config_path) if config_path else config or {}
        self.runtime_config = runtime_config or {}
        
        # 保存 scene 文件所在的目录（用于相对路径转换）
        self.scene_dir = None  # 在 generate_complete_scene 中设置
        
        # 缓存 SPH body 的 geom 信息，避免重复查找
        # 格式: {body_name: GeomInfo}
        self._sph_geom_cache = {}
        
        # 基准目录：scene_generator.py 所在目录（用于解析配置文件中的相对路径）
        self._base_dir = os.path.dirname(os.path.abspath(__file__))
        
        logger.info(f"SceneGenerator initialized")
    
    def _resolve_geometry_path(self, geometry_file: str) -> str:
        """
        将相对路径转换为绝对路径，支持包资源加载
        
        支持三种路径格式：
        1. 绝对路径：直接返回
        2. package://orcasph/data/models/UnitBox.obj：从 orcasph_client 包加载
        3. 相对路径：相对于 scene_generator.py 所在目录，如果不存在则尝试从包中加载
        
        Args:
            geometry_file: 几何文件路径（可能是相对路径或绝对路径）
            
        Returns:
            绝对路径
        """
        if not geometry_file:
            return geometry_file
            
        # 如果已经是绝对路径，直接返回
        if os.path.isabs(geometry_file):
            return geometry_file
        
        # 支持包资源路径：package://orcasph/data/models/UnitBox.obj
        if geometry_file.startswith('package://orcasph/'):
            resource_path = geometry_file[len('package://orcasph/'):]
            abs_path = self._load_from_orcasph_package(resource_path)
            if abs_path:
                logger.info(f"Loaded from orcasph package: '{geometry_file}' -> '{abs_path}'")
                return abs_path
            else:
                logger.warning(f"Failed to load package resource: '{geometry_file}'")
        
        # 相对路径：从 scene_generator.py 所在目录解析
        abs_path = os.path.abspath(os.path.join(self._base_dir, geometry_file))
        
        # 如果文件不存在，尝试从 orcasph_client 包中加载（fallback）
        if not os.path.exists(abs_path):
            logger.warning(f"File not found at: {abs_path}")
            
            # 尝试从包中加载
            # 例如：../../../data/models/UnitBox.obj -> data/models/UnitBox.obj
            filename = os.path.basename(geometry_file)
            dirname = os.path.dirname(geometry_file)
            
            # 尝试常见的资源路径
            possible_paths = [
                f"data/models/{filename}",
                f"data/{filename}",
                filename,
            ]
            
            for resource_path in possible_paths:
                package_path = self._load_from_orcasph_package(resource_path)
                if package_path and os.path.exists(package_path):
                    logger.info(f"Found in orcasph package: '{geometry_file}' -> '{package_path}'")
                    return package_path
            
            logger.warning(f"Could not find file in package, using original path: {abs_path}")
        
        logger.info(f"Resolved geometry path: '{geometry_file}' -> '{abs_path}'")
        return abs_path
    
    def _load_from_orcasph_package(self, resource_path: str) -> str:
        """
        从 orcasph_client 包中加载资源文件
        
        兼容：
        - pip install orca-sph（普通安装）
        - pip install -e .（可编辑安装）
        
        Args:
            resource_path: 包内资源路径，如 'data/models/UnitBox.obj'
            
        Returns:
            资源文件的绝对路径，如果找不到返回 None
        """
        try:
            # 方法1：使用 __file__ 定位（兼容 editable install）
            try:
                import orcasph_client
                package_dir = Path(orcasph_client.__file__).parent
                resource_file = package_dir / resource_path
                
                if resource_file.exists():
                    return str(resource_file.resolve())
            except ImportError:
                pass
            
            # 方法2：使用 importlib.resources（Python 3.9+）
            try:
                from importlib import resources
                if hasattr(resources, 'files'):
                    resource = resources.files('orcasph_client').joinpath(resource_path)
                    if resource.is_file():
                        return str(resource)
            except Exception:
                pass
            
            # 方法3：使用 pkg_resources（fallback，兼容旧版本）
            try:
                import pkg_resources
                resource_file = pkg_resources.resource_filename('orcasph_client', resource_path)
                if os.path.exists(resource_file):
                    return resource_file
            except Exception:
                pass
            
            return None
            
        except Exception as e:
            logger.debug(f"Error loading package resource '{resource_path}': {e}")
            return None
    
    def _load_config(self, config_path: str) -> Dict:
        """加载 JSON 配置文件"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return {}
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in config file: {e}, using defaults")
            return {}
    
    def identify_sph_bodies(self) -> List[str]:
        """
        识别需要导出的刚体（有 SPH_MESH_GEOM 或 SPH_STATIC_MESH_GEOM 的 body）并缓存 geom 信息
        
        识别策略：
        1. 遍历所有 geom，根据名称过滤出 SPH_MESH_GEOM 和 SPH_STATIC_MESH_GEOM
        2. 从 geom 的 BodyName 字段直接获取 body 名称
        3. 同时获取 mesh 文件路径和 scale 信息并缓存
        4. 去重并返回
        
        命名规则：
        - 动态刚体：${bodyName}_SPH_MESH_GEOM，对应 mesh 为 ${bodyName}_SPH_MESH
        - 静态刚体：${bodyName}_SPH_STATIC_MESH_GEOM，对应 mesh 为 ${bodyName}_SPH_STATIC_MESH
        
        Returns:
            List[str]: body 名称列表
        """
        sph_bodies = set()
        self._sph_geom_cache.clear()  # 清空缓存
        
        try:
            model = self.env.model
            
            # 获取 geom 字典和 mesh 字典
            geom_dict = model.get_geom_dict()
            mesh_dict = model.get_mesh_dict()
            
            if not geom_dict:
                logger.warning("geom_dict not available")
                return []
            
            logger.info(f"Scanning {len(geom_dict)} geoms for SPH geoms (dynamic and static)...")
            
            # 过滤出 SPH_MESH_GEOM 和 SPH_STATIC_MESH_GEOM 并提取信息
            sph_geoms = []
            for geom_name, geom_info in geom_dict.items():
                # 检查是否为静态刚体或动态刚体
                is_static = '_SPH_STATIC_MESH_GEOM' in geom_name
                is_dynamic = '_SPH_MESH_GEOM' in geom_name and not is_static
                
                if is_static or is_dynamic:
                    body_name = geom_info.get('BodyName', '')
                    if not body_name:
                        logger.warning(f"SPH geom '{geom_name}' has no BodyName!")
                        continue
                    
                    sph_bodies.add(body_name)
                    sph_geoms.append({
                        'geom': geom_name, 
                        'body': body_name, 
                        'is_static': is_static
                    })
                    
                    # 获取并缓存 geom 信息（mesh 文件路径和 scale）
                    # 注意：
                    # 1. 不能通过 geom 的 DataID 获取，因为 DataID 可能指向其他用途的 mesh
                    # 2. 应该直接通过命名规则获取 SPH_MESH 或 SPH_STATIC_MESH
                    # 3. scale 从 mesh 的 Scale 属性获取（从 XML 解析）
                    try:
                        # 根据类型构造 SPH_MESH 或 SPH_STATIC_MESH 名称
                        if is_static:
                            sph_mesh_name = f"{body_name}_SPH_STATIC_MESH"
                        else:
                            sph_mesh_name = f"{body_name}_SPH_MESH"
                        
                        # 通过 mesh 名称获取 mesh 信息
                        mesh_info = mesh_dict.get(sph_mesh_name) if mesh_dict else None
                        
                        if mesh_info:
                            mesh_file = mesh_info.get('File', '')
                            # mesh scale 从 XML 中解析得到
                            mesh_scale = mesh_info.get('Scale', [1.0, 1.0, 1.0])
                            
                            # 缓存 GeomInfo
                            self._sph_geom_cache[body_name] = GeomInfo(
                                geom_type='mesh',
                                mesh_name=mesh_file,
                                size=[],
                                scale=list(mesh_scale) if mesh_scale else [1.0, 1.0, 1.0],
                                is_static=is_static
                            )
                            
                            body_type = "static" if is_static else "dynamic"
                            logger.info(f"Cached {body_type} geom info for '{body_name}': mesh='{sph_mesh_name}', file='{mesh_file}', scale={mesh_scale}")
                        else:
                            logger.warning(f"SPH_MESH '{sph_mesh_name}' not found in mesh_dict for body '{body_name}'")
                    except Exception as e:
                        logger.warning(f"Failed to cache geom info for '{body_name}': {e}", exc_info=True)
            
            # 日志输出识别结果
            if sph_geoms:
                dynamic_count = sum(1 for g in sph_geoms if not g['is_static'])
                static_count = sum(1 for g in sph_geoms if g['is_static'])
                logger.info(f"Found {len(sph_geoms)} SPH geom(s): {dynamic_count} dynamic, {static_count} static")
                for item in sph_geoms:
                    body_type = "static" if item['is_static'] else "dynamic"
                    logger.info(f"  - [{body_type}] geom: '{item['geom']}' -> body: '{item['body']}'")
            
            result = sorted(list(sph_bodies))
            logger.info(f"Identified {len(result)} SPH body(bodies): {result}")
            logger.info(f"Cached geom info for {len(self._sph_geom_cache)} bodies")
            return result
            
        except Exception as e:
            logger.error(f"Error identifying SPH bodies: {e}", exc_info=True)
            return []
    
    def extract_mocap_bodies_for_body(self, body_name: str) -> List[Dict]:
        """
        提取指定主刚体对应的所有 Mocap site 世界坐标位置
        
        识别策略：
        1. 遍历所有 site
        2. 查找包含 "SPH_MOCAP_SITE" 且属于该 body 的 site
        3. 使用 query_site_pos_and_mat 读取世界坐标位置
        
        Args:
            body_name: 主刚体名称，例如 "toys_usda_box_body"
        
        Returns:
            List of dicts with keys:
                - 'mocap_body_name': 完整的 mocap body 名称
                - 'index': mocap 索引 (0-3)
                - 'world_position': 世界坐标位置 [x, y, z] (MuJoCo Z-up 坐标系)
        """
        mocap_info_list = []
        
        try:
            model = self.env.model
            
            # 1. 获取所有 site 的字典
            site_dict = model.get_site_dict()
            if not site_dict:
                logger.warning(f"site_dict not available for body '{body_name}'")
                return []
            
            # 2. 筛选出属于该主刚体的 mocap site
            # 命名模式: "{body_name}_SPH_MOCAP_SITE_{index:03d}"
            prefix = f"{body_name}_SPH_MOCAP_SITE_"
            
            # 收集符合条件的 site 名称
            site_names = []
            for site_name in site_dict.keys():
                if site_name.startswith(prefix):
                    site_names.append(site_name)
            
            if not site_names:
                logger.debug(f"No SPH_MOCAP_SITE found for body '{body_name}'")
                return []
            
            # 3. 使用 query_site_pos_and_mat 获取世界坐标（需要先调用 mj_forward 更新数据）
            self.env.mj_forward()
            site_pos_data = self.env.query_site_pos_and_mat(site_names)
            
            # 4. 构造返回列表，包含 world_position
            for site_name in site_names:
                # 提取索引
                try:
                    index_str = site_name.replace(prefix, "")
                    index = int(index_str)
                except ValueError:
                    continue
                
                # 获取世界坐标位置
                if site_name in site_pos_data:
                    xpos = site_pos_data[site_name]['xpos']
                    # 转换为列表格式
                    if isinstance(xpos, np.ndarray):
                        world_pos = xpos[:3].tolist()
                    elif isinstance(xpos, (list, tuple)) and len(xpos) >= 3:
                        world_pos = list(xpos[:3])
                    else:
                        logger.warning(f"Invalid xpos format for site '{site_name}': {xpos}")
                        world_pos = [0.0, 0.0, 0.0]
                else:
                    logger.warning(f"Site '{site_name}' not found in query_site_pos_and_mat result")
                    world_pos = [0.0, 0.0, 0.0]
                
                # 构造对应的 mocap body 名称
                mocap_body_name = f"{body_name}_SPH_MOCAP_{index:03d}"
                
                mocap_info_list.append({
                    'mocap_body_name': mocap_body_name,
                    'index': index,
                    'world_position': world_pos  # MuJoCo 世界坐标 (Z-up)
                })
            
            # 5. 按索引排序
            mocap_info_list.sort(key=lambda x: x['index'])
            
            logger.debug(f"Extracted {len(mocap_info_list)} mocap bodies for '{body_name}'")
            return mocap_info_list
            
        except Exception as e:
            logger.error(f"Error extracting mocap bodies for '{body_name}': {e}", exc_info=True)
            return []
    
    def convert_local_coord_z_to_y(self, local_pos_mj: np.ndarray) -> List[float]:
        """
        转换本地坐标从 MuJoCo Z-up 到 SPH Y-up
        
        坐标系转换规则：
            MuJoCo Z-up: X right, Y forward, Z up
            SPH Y-up:    X right, Z forward, Y up
        
        转换公式：[x, y, z] → [x, z, -y]
        
        Args:
            local_pos_mj: MuJoCo 本地坐标 [x, y, z]
        
        Returns:
            SPH 本地坐标 [x, z, -y]
        """
        return [float(local_pos_mj[0]), float(local_pos_mj[2]), -float(local_pos_mj[1])]
    
    def extract_site_local_positions_for_body(self, body_name: str) -> List[Dict]:
        """
        提取指定主刚体的所有 SPH_SITE 本地坐标位置
        
        Args:
            body_name: 主刚体名称
        
        Returns:
            List of dicts:
                - 'site_name': 完整的 site 名称
                - 'index': site 索引
                - 'local_position': 本地坐标 [x, y, z] (MuJoCo Z-up 坐标系)
        """
        site_info_list = []
        
        try:
            # 从 model 获取所有 site 的字典（包含 LocalPos）
            site_dict = self.env.model.get_site_dict()
            prefix = f"{body_name}_SPH_SITE_"
            
            for site_name, site_data in site_dict.items():
                if site_name.startswith(prefix):
                    # 提取索引
                    try:
                        index_str = site_name.replace(prefix, "")
                        index = int(index_str)
                    except ValueError:
                        continue
                    
                    # 直接读取本地坐标（已经是相对于 body 的坐标）
                    local_pos = site_data.get('LocalPos', np.array([0.0, 0.0, 0.0]))
                    if isinstance(local_pos, np.ndarray):
                        local_pos = local_pos.tolist()
                    elif isinstance(local_pos, (list, tuple)):
                        local_pos = list(local_pos[:3])
                    else:
                        logger.warning(f"Invalid LocalPos format for site '{site_name}': {local_pos}")
                        local_pos = [0.0, 0.0, 0.0]
                    
                    site_info_list.append({
                        'site_name': site_name,
                        'index': index,
                        'local_position': local_pos  # MuJoCo 本地坐标 (Z-up)
                    })
            
            site_info_list.sort(key=lambda x: x['index'])
            logger.debug(f"Extracted {len(site_info_list)} site local positions for '{body_name}'")
            return site_info_list
            
        except Exception as e:
            logger.error(f"Error extracting site local positions for '{body_name}': {e}", exc_info=True)
            return []
    
    def extract_mocap_world_positions_for_body(self, body_name: str) -> List[Dict]:
        """
        提取指定主刚体的所有 SPH_MOCAP_SITE 全局坐标位置
        
        注意：MOCAP body 是独立的 mocap body，不隶属于主刚体
        需要从 body_xpos 读取全局位置
        
        Args:
            body_name: 主刚体名称
        
        Returns:
            List of dicts:
                - 'mocap_name': 完整的 mocap body 名称
                - 'index': mocap body 索引
                - 'world_position': 全局坐标 [x, y, z] (SPH Y-up 坐标系)
        """
        mocap_info_list = []
        
        try:
            # 1. 从 model 获取所有 body 的字典，筛选出属于该主刚体的 SPH_MOCAP body
            body_dict = self.env.model.get_body_dict()
            prefix = f"{body_name}_SPH_MOCAP_"
            
            # 收集符合条件的 mocap body 名称
            mocap_body_names = []
            for body_name_mj in body_dict.keys():
                if body_name_mj.startswith(prefix):
                    mocap_body_names.append(body_name_mj)
            
            if not mocap_body_names:
                logger.debug(f"No SPH_MOCAP body found for '{body_name}'")
                return []
            
            # 2. 使用 get_body_xpos_xmat_xquat 获取世界坐标（需要先调用 mj_forward 更新数据）
            self.env.mj_forward()
            xpos_flat, _, _ = self.env.get_body_xpos_xmat_xquat(mocap_body_names)
            
            # 3. 构造返回列表，包含 world_position (MuJoCo 坐标系，未转换)
            for i, mocap_body_name in enumerate(mocap_body_names):
                # 提取索引
                try:
                    index_str = mocap_body_name.replace(prefix, "")
                    index = int(index_str)
                except ValueError:
                    continue
                
                # 获取世界坐标位置（xpos_flat 是扁平化的数组，每个 body 3个元素）
                world_pos_mj = xpos_flat[i*3:(i+1)*3]
                
                mocap_info_list.append({
                    'mocap_name': mocap_body_name,
                    'index': index,
                    'world_position': world_pos_mj.tolist() if isinstance(world_pos_mj, np.ndarray) else list(world_pos_mj)
                })
            
            mocap_info_list.sort(key=lambda x: x['index'])
            logger.debug(f"Extracted {len(mocap_info_list)} mocap world positions for '{body_name}'")
            return mocap_info_list
            
        except Exception as e:
            logger.error(f"Error extracting mocap world positions for '{body_name}': {e}", exc_info=True)
            return []
    
    def extract_site_positions_for_body(self, body_name: str) -> List[Dict]:
        """
        提取指定主刚体的所有 SPH_SITE 世界坐标位置
        
        Args:
            body_name: 主刚体名称
        
        Returns:
            List of dicts with keys:
                - 'site_name': 完整的 site 名称
                - 'index': site 索引 (0-3)
                - 'world_position': 世界坐标位置 [x, y, z] (MuJoCo Z-up 坐标系)
        """
        site_info_list = []
        
        try:
            # 1. 从 model 获取所有 site 的字典，筛选出属于该主刚体的 SPH_SITE
            site_dict = self.env.model.get_site_dict()
            prefix = f"{body_name}_SPH_SITE_"
            
            # 收集符合条件的 site 名称
            site_names = []
            for site_name in site_dict.keys():
                if site_name.startswith(prefix):
                    site_names.append(site_name)
            
            if not site_names:
                logger.debug(f"No SPH_SITE found for body '{body_name}'")
                return []
            
            # 2. 使用 query_site_pos_and_mat 获取世界坐标（需要先调用 mj_forward 更新数据）
            self.env.mj_forward()
            site_pos_data = self.env.query_site_pos_and_mat(site_names)
            
            # 3. 构造返回列表，包含 world_position
            for site_name in site_names:
                # 提取索引
                try:
                    index_str = site_name.replace(prefix, "")
                    index = int(index_str)
                except ValueError:
                    continue
                
                # 获取世界坐标位置
                if site_name in site_pos_data:
                    xpos = site_pos_data[site_name]['xpos']
                    # 转换为列表格式
                    if isinstance(xpos, np.ndarray):
                        world_pos = xpos[:3].tolist()
                    elif isinstance(xpos, (list, tuple)) and len(xpos) >= 3:
                        world_pos = list(xpos[:3])
                    else:
                        logger.warning(f"Invalid xpos format for site '{site_name}': {xpos}")
                        world_pos = [0.0, 0.0, 0.0]
                else:
                    logger.warning(f"Site '{site_name}' not found in query_site_pos_and_mat result")
                    world_pos = [0.0, 0.0, 0.0]
                
                site_info_list.append({
                    'site_name': site_name,
                    'index': index,
                    'world_position': world_pos  # MuJoCo 世界坐标 (Z-up)
                })
            
            # 4. 按索引排序
            site_info_list.sort(key=lambda x: x['index'])
            
            logger.debug(f"Extracted {len(site_info_list)} site positions for '{body_name}'")
            return site_info_list
            
        except Exception as e:
            logger.error(f"Error extracting site positions for '{body_name}': {e}", exc_info=True)
            return []
    
    def extract_body_info(self, body_name: str) -> Dict:
        """
        从 OrcaGym model 提取 body 完整信息
        
        Args:
            body_name: MuJoCo body 名称
            
        Returns:
            Dict: body 信息字典
        """
        try:
            model = self.env.model
            
            # 获取 body ID
            body_id = model.body_name2id(body_name)
            
            # 获取位置和旋转 - 返回扁平数组，需要提取单个 body 数据
            xpos_flat, _, xquat_flat = self.env.get_body_xpos_xmat_xquat([body_name])
            
            # 提取单个 body 的数据
            if xpos_flat is not None and len(xpos_flat) >= 3:
                xpos = np.array(xpos_flat[:3])
            else:
                xpos = np.array([0.0, 0.0, 0.0])
            
            if xquat_flat is not None and len(xquat_flat) >= 4:
                xquat = np.array(xquat_flat[:4])
            else:
                xquat = np.array([1.0, 0.0, 0.0, 0.0])
            
            # 获取质量
            body_dict = model.get_body_dict()
            body_data = body_dict.get(body_name, {})
            mass = body_data.get('Mass', 1.0)
            subtree_mass = body_data.get('SubtreeMass', 1.0)
            inertia = body_data.get('Inertia', [0, 0, 0])
            
            # 调试：打印质量相关信息
            logger.info(f"[DEBUG] Body '{body_name}' mass info:")
            logger.info(f"  Mass: {mass}")
            logger.info(f"  SubtreeMass: {subtree_mass}")
            logger.info(f"  Inertia: {inertia}")
            
            # 从缓存获取 geom 信息（在 identify_sph_bodies 中已经获取并缓存）
            geom_info = self._sph_geom_cache.get(body_name)
            if geom_info is None:
                error_msg = (
                    f"Geom info not found in cache for body '{body_name}'. "
                    f"Please ensure identify_sph_bodies() is called before extract_body_info()."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            logger.debug(f"Extracted body info for '{body_name}': pos={xpos}, quat={xquat}, mass={mass}")
            
            return {
                'body_id': body_id,
                'body_name': body_name,
                'position': xpos,
                'quaternion': xquat,
                'mass': mass,
                'geom_info': geom_info
            }
            
        except Exception as e:
            error_msg = f"Error extracting body info for '{body_name}': {e}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
    
    def convert_coordinate_z_to_y(self, pos: np.ndarray, quat: np.ndarray) -> Tuple[List, List, float]:
        """
        坐标系转换：MuJoCo Z-up → SPH Y-up
        
        坐标系定义：
            MuJoCo Z-up: X right, Y forward, Z up
            SPH Y-up:    X right, Z forward, Y up
            
        转换公式：
            位置: [x, y, z] → [x, z, -y]
            旋转: 绕 X 轴旋转 -90 度
        
        Args:
            pos: 位置 [x, y, z]
            quat: 四元数 [w, x, y, z]
            
        Returns:
            Tuple: (translation, rotationAxis, rotationAngle)
        """
        try:
            from scipy.spatial.transform import Rotation as R
            
            # 位置转换：MuJoCo Z-up [x, y, z] → SPH Y-up [x, z, -y]
            translation = [float(pos[0]), float(pos[2]), -float(pos[1])]
            
            # 四元数转换（MuJoCo 格式 [w, x, y, z] → SciPy 格式 [x, y, z, w]）
            rot = R.from_quat([float(quat[1]), float(quat[2]), float(quat[3]), float(quat[0])])
            
            # 应用坐标系变换矩阵（绕X轴旋转-90度）
            # 从 Z-up 到 Y-up：Y → -Z, Z → Y，相当于绕 X 轴旋转 -90 度
            transform = R.from_euler('x', -90, degrees=True)
            rot_transformed = transform * rot
            
            # 转换为轴角表示
            rotvec = rot_transformed.as_rotvec()
            angle = float(np.linalg.norm(rotvec))
            
            if angle > 1e-6:
                axis = (rotvec / angle).tolist()
            else:
                axis = [1.0, 0.0, 0.0]
            
            logger.debug(f"Coordinate converted: pos={pos} → {translation}, quat={quat} → axis={axis}, angle={angle}")
            
            return translation, axis, angle
            
        except Exception as e:
            logger.error(f"Error in coordinate conversion: {e}", exc_info=True)
            # 返回默认值
            return [float(pos[0]), float(pos[2]), -float(pos[1])], [1.0, 0.0, 0.0], 0.0
    
    def _extract_entity_name(self, body_name: str) -> str:
        """
        从 MuJoCo body 名称提取 entityName
        
        直接使用原始 body 名称，无需转换
        (body name 设计满足 Python 变量约束格式)
        
        例如：
            "toys_usda_sphere_body" → "toys_usda_sphere_body"
            "toys_usda_box_body" → "toys_usda_box_body"
        
        Args:
            body_name: MuJoCo body 名称
            
        Returns:
            str: entityName（与 body_name 相同）
        """
        return body_name
    
    def extract_rigid_body_config(self, body_name: str, rb_id: int) -> Dict:
        """
        提取单个刚体的完整配置
        
        Args:
            body_name: MuJoCo body 名称
            rb_id: �体在 scene.json 中的 ID
            
        Returns:
            Dict: 刚体配置（SPH scene.json 格式）
        """
        try:
            body_info = self.extract_body_info(body_name)
            
            # 坐标转换
            translation, rot_axis, rot_angle = self.convert_coordinate_z_to_y(
                body_info['position'],
                body_info['quaternion']
            )
            
            # 生成 entityName
            entity_name = self._extract_entity_name(body_name)
            
            # 获取默认值
            default_rb = self.config.get('default_rigid_body', {})
            
            # 根据 geom_info.is_static 确定 isDynamic
            is_static = body_info['geom_info'].is_static
            is_dynamic = not is_static if is_static else default_rb.get('isDynamic', True)
            
            config = {
                "id": rb_id,
                "entityName": entity_name,
                "geometryFile": body_info['geom_info'].mesh_name,
                "isDynamic": is_dynamic,
                "density": default_rb.get('density', 500),  # 保留作为默认值
                "mass": body_info['mass'],  # 新增：从 MuJoCo 读取的质量
                "translation": translation,
                "rotationAxis": rot_axis,
                "rotationAngle": rot_angle,
                "scale": body_info['geom_info'].scale if body_info['geom_info'].scale else [1, 1, 1],
                "velocity": [0, 0, 0],
                "restitution": default_rb.get('restitution', 0.5),
                "friction": default_rb.get('friction', 0.25),
                "color": [0.5, 0.5, 0.5, 1.0],
                "collisionObjectType": 2,  # 固定为 Box（刚体碰撞关闭时不影响）
                "collisionObjectScale": body_info['geom_info'].scale if body_info['geom_info'].scale else [1.0, 1.0, 1.0],
                "mapInvert": default_rb.get('mapInvert', False),
                "mapThickness": default_rb.get('mapThickness', 0.0),
                "mapResolution": default_rb.get('mapResolution', [20, 20, 20])
            }
            
            body_type = "static" if is_static else "dynamic"
            logger.debug(f"Generated {body_type} rigid body config for '{body_name}': {config}")
            return config
            
        except Exception as e:
            logger.error(f"Error extracting rigid body config for '{body_name}': {e}", exc_info=True)
            raise
    
    def generate_anchor_points(self, main_body_name: str, rb_id: int) -> Optional[Dict]:
        """
        为指定刚体生成 AnchorPoints 配置
        
        使用世界坐标，通过 query_site_pos_and_mat API 读取 SITE 点世界坐标
        
        Args:
            main_body_name: 主刚体名称
            rb_id: 刚体 ID
        
        Returns:
            Dict with anchor points configuration or None if no points found
        """
        # 1. 提取 site 点世界坐标（虚拟锚点）
        site_infos = self.extract_site_positions_for_body(main_body_name)
        site_points = []
        for site in site_infos:
            world_pos_mj = np.array(site['world_position'])
            # 坐标系转换 Z-up → Y-up
            world_pos_sph = self.convert_local_coord_z_to_y(world_pos_mj)
            site_points.append({
                "point_id": site['site_name'],
                "initial_world_pos": world_pos_sph
            })
        
        # 2. 提取 mocap 点全局坐标（牵引点，独立的 mocap body）
        mocap_infos = self.extract_mocap_world_positions_for_body(main_body_name)
        mocap_points = []
        for mocap in mocap_infos:
            world_pos_mj = np.array(mocap['world_position'])
            # 坐标系转换 Z-up → Y-up (和 site 统一处理)
            world_pos_sph = self.convert_local_coord_z_to_y(world_pos_mj)
            mocap_points.append({
                "point_id": mocap['mocap_name'],
                "world_pos": world_pos_sph  # 全局坐标（已转换为 SPH Y-up）
            })
        
        if not site_points and not mocap_points:
            return None
        
        # 3. 从运行时配置读取弹簧参数（必需配置）
        spring_force_config = (
            self.runtime_config
            .get('orcalink_bridge', {})
            .get('shared_modules', {})
            .get('spring_force', {})
        )

        if not spring_force_config:
            raise ValueError(
                "弹簧力配置缺失！必须在 sph_sim_config.json 中配置 "
                "'orcalink_bridge.shared_modules.spring_force' 字段"
            )

        spring_stiffness = spring_force_config.get('linear_spring_stiffness')
        spring_damping = spring_force_config.get('linear_damping_coefficient')

        if spring_stiffness is None or spring_damping is None:
            raise ValueError(
                f"弹簧参数不完整！需要配置:\n"
                f"  - linear_spring_stiffness (当前: {spring_stiffness})\n"
                f"  - linear_damping_coefficient (当前: {spring_damping})\n"
                f"请在 sph_sim_config.json 的 orcalink_bridge.shared_modules.spring_force 中配置"
            )

        logger.info(
            f"[SceneGenerator] 使用弹簧参数: k={spring_stiffness} N/m, c={spring_damping} N·s/m"
        )
        
        return {
            "rigid_body_id": rb_id,
            "object_name": main_body_name,
            "spring_stiffness": spring_stiffness,
            "spring_damping": spring_damping,
            "site_points": site_points,
            "mocap_points": mocap_points
        }
    
    
    def generate_scene_json(self, output_path: str = None) -> Dict:
        """
        生成完整的 scene.json RigidBodies 部分
        
        Args:
            output_path: 输出文件路径（可选）
            
        Returns:
            Dict: RigidBodies 配置字典
        """
        try:
            # 识别 SPH bodies
            sph_bodies = self.identify_sph_bodies()
            
            if not sph_bodies:
                logger.warning("No SPH bodies found in the model")
                return {"RigidBodies": []}
            
            # 在读取 data 数据前，先调用 mj_forward 更新数据
            self.env.mj_forward()
            
            # 提取每个 body 的配置
            rigid_bodies = []
            for rb_id, body_name in enumerate(sph_bodies, start=1):
                config = self.extract_rigid_body_config(body_name, rb_id)
                rigid_bodies.append(config)
            
            scene_data = {
                "Configuration": self.config.get('scene_template', {}).get('Configuration', {
                    "particleRadius": 0.025,
                    "numberOfStepsPerRenderUpdate": 4,
                    "density0": 1000,
                    "simulationMethod": 4,
                    "gravitation": [0, -9.81, 0],
                    "cflMethod": 1,
                    "cflFactor": 1,
                    "cflMaxTimeStepSize": 0.005,
                    "maxIterations": 100,
                    "maxError": 0.05,
                    "maxIterationsV": 100,
                    "maxErrorV": 0.1,
                    "stiffness": 50000,
                    "exponent": 7,
                    "velocityUpdateMethod": 0,
                    "enableDivergenceSolver": True,
                    "particleAttributes": "velocity",
                    "boundaryHandlingMethod": 2
                }),
                "Materials": self.config.get('scene_template', {}).get('Materials', [
                    {
                        "id": "Fluid",
                        "colorMapType": 1,
                        "surfaceTension": 0.2,
                        "surfaceTensionMethod": 0,
                        "viscosity": 0.01,
                        "viscosityMethod": 1
                    }
                ]),
                "RigidBodies": rigid_bodies
            }
            
            # 保存到文件（如果指定了路径）
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w') as f:
                    json.dump(scene_data, f, indent=4)
                logger.info(f"Scene JSON generated and saved to {output_path}")
            
            logger.info(f"Generated scene with {len(rigid_bodies)} rigid bodies")
            return scene_data
            
        except Exception as e:
            logger.error(f"Error generating scene JSON: {e}", exc_info=True)
            raise
    
    def generate_complete_scene(self, output_path: str = None, 
                               include_fluid_blocks: bool = True,
                               include_wall: bool = True) -> Dict:
        """
        生成完整的 scene.json 文件（包括 Configuration、Materials、RigidBodies、FluidBlocks）
        
        新增功能：
        1. 为每个主刚体生成 Site 可视化刚体
        2. 为每个主刚体生成 Mocap 可视化刚体
        3. 生成 FixedJoint 约束（Site -> Main）
        4. 生成 RigidBodySpring 约束（Mocap -> Site）
        
        Args:
            output_path: 输出文件路径（可选）
            include_fluid_blocks: 是否包含 FluidBlocks 模板（默认 True）
            include_wall: 是否包含容器墙体（默认 True）
            
        Returns:
            Dict: 完整的场景配置字典
        """
        try:
            # 设置 scene 目录（用于 SPH_MESH 文件的相对路径转换）
            if output_path:
                self.scene_dir = str(Path(output_path).parent.absolute())
                logger.info(f"Scene directory set to: {self.scene_dir}")
            
            # 在读取 data 数据前，先调用 mj_forward 更新数据
            self.env.mj_forward()
            
            # 生成主刚体
            main_rigid_bodies = self.generate_scene_json(output_path=None)["RigidBodies"]
            
            # 准备收集所有刚体
            all_rigid_bodies = []
            
            # 当前 ID 计数器
            current_rb_id = 0
            
            # Wall container (id=0)
            if include_wall:
                wall_config = self.config.get('wall_rigid_body', {})
                if wall_config:
                    # 转换 geometryFile 为绝对路径
                    geometry_file = wall_config.get('geometryFile', "../models/UnitBox.obj")
                    geometry_file = self._resolve_geometry_path(geometry_file)
                    
                    wall_rigid_body = {
                        "id": current_rb_id,
                        "geometryFile": geometry_file,
                        "translation": wall_config.get('translation', [0, 3.0, 0]),
                        "rotationAxis": wall_config.get('rotationAxis', [1, 0, 0]),
                        "rotationAngle": wall_config.get('rotationAngle', 0),
                        "scale": wall_config.get('scale', [1.5, 6, 1.5]),
                        "color": wall_config.get('color', [0.1, 0.4, 0.6, 1.0]),
                        "isDynamic": wall_config.get('isDynamic', False),
                        "isWall": wall_config.get('isWall', True),
                        "collisionObjectType": wall_config.get('collisionObjectType', 2),
                        "collisionObjectScale": wall_config.get('collisionObjectScale', [1.5, 6, 1.5]),
                        "invertSDF": wall_config.get('invertSDF', True),
                        "mapInvert": wall_config.get('mapInvert', True),
                        "mapThickness": wall_config.get('mapThickness', 0.0),
                        "mapResolution": wall_config.get('mapResolution', [30, 50, 30])
                    }
                    all_rigid_bodies.append(wall_rigid_body)
                    current_rb_id += 1
            
            # 为每个主刚体添加（不再生成辅助刚体，改用虚拟锚点粒子方案）
            for main_rb in main_rigid_bodies:
                main_body_name = main_rb.get("entityName")
                if not main_body_name:
                    logger.warning(f"Main rigid body missing entityName, skipping: {main_rb}")
                    continue
                
                # 添加主刚体
                main_rb["id"] = current_rb_id
                all_rigid_bodies.append(main_rb)
                current_rb_id += 1
                
                # 提取 site 信息用于后续虚拟锚点配置（不生成辅助刚体）
                site_infos = self.extract_site_positions_for_body(main_body_name)
                if site_infos:
                    logger.debug(f"Found {len(site_infos)} sites for '{main_body_name}' (will be used for anchor particles)")
            
            # 组装完整场景
            scene_template = self.config.get('scene_template', {})
            complete_scene = {
                "Configuration": scene_template.get("Configuration", {}),
                "Materials": scene_template.get("Materials", [])
            }
            
            # 添加刚体
            complete_scene["RigidBodies"] = all_rigid_bodies
            
            # 生成 AnchorPoints
            logger.info("Generating AnchorPoints configuration...")
            anchor_points = []
            for main_rb in main_rigid_bodies:
                main_body_name = main_rb.get("entityName")
                if not main_body_name:
                    continue
                rb_id = main_rb.get("id")
                if rb_id is None:
                    continue
                
                # 跳过静态刚体（不需要锚点）
                if main_rb.get("isDynamic") == False:
                    logger.info(f"  Skipping anchor points for static body '{main_body_name}'")
                    continue
                
                anchor_point = self.generate_anchor_points(main_body_name, rb_id)
                if anchor_point:
                    anchor_points.append(anchor_point)
                    logger.info(f"  Generated {len(anchor_point['site_points'])} site points + "
                               f"{len(anchor_point['mocap_points'])} mocap points for '{main_body_name}'")
            
            # 添加到场景
            if anchor_points:
                complete_scene["AnchorPoints"] = anchor_points
                logger.info(f"Total: {len(anchor_points)} rigid bodies with anchor points")
            
            # 注意：不再生成辅助刚体和约束，虚拟锚点粒子将在运行时通过 PBD 创建
            
            # 添加 FluidBlocks
            if include_fluid_blocks:
                complete_scene["FluidBlocks"] = scene_template.get('FluidBlocks', [
                    {
                        "denseMode": 0,
                        "start": [-0.35, -0.35, -0.35],
                        "end": [0.35, 0.35, 0.35],
                        "translation": [0.0, 0.5, 0.0],
                        "scale": [2.0, 1.0, 2.0]
                    }
                ])
            
            # 保存到文件（如果指定了路径）
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w') as f:
                    json.dump(complete_scene, f, indent=2)
                logger.info(f"Complete scene with anchor system saved to {output_path}")
            
            logger.info(f"Generated complete scene with {len(all_rigid_bodies)} rigid bodies "
                       f"(virtual anchor particles will be created at runtime via PBD)")
            return complete_scene
            
        except Exception as e:
            logger.error(f"Error generating complete scene: {e}", exc_info=True)
            raise


# 便捷函数
def generate_scene_from_env(env, output_path: str = None, config_path: str = None) -> Dict:
    """
    便捷函数：从 OrcaGym 环境生成 scene.json
    
    Args:
        env: OrcaGymLocalEnv 实例
        output_path: 输出文件路径
        config_path: 配置文件路径
        
    Returns:
        Dict: 生成的 scene 配置
    """
    generator = SceneGenerator(env, config_path=config_path)
    return generator.generate_scene_json(output_path)

