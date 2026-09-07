"""
SPH-MuJoCo 配置生成器

从 MuJoCo 模型动态生成 sph_mujoco_config.json 的 rigid_bodies 部分。
自动识别 SPH_SITE 和 SPH_MOCAP_SITE，生成连接点配置。
"""

import logging
from typing import Dict, List
import re

from .body_name_map import resolve_internal_prefix

# 设置日志
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('[ConfigGenerator] %(levelname)s: %(message)s'))
    logger.addHandler(handler)


class ConfigGenerator:
    """SPH-MuJoCo 配置生成器"""
    
    def __init__(self, env):
        """
        初始化配置生成器
        
        Args:
            env: OrcaGymLocalEnv 实例
        """
        import sys
        print("[PRINT-DEBUG] ConfigGenerator.__init__() - START", file=sys.stderr, flush=True)
        self.env = env
        self.model = env.model
        print("[PRINT-DEBUG] ConfigGenerator.__init__() - END", file=sys.stderr, flush=True)
    
    def identify_sph_bodies(self) -> List[str]:
        """
        识别所有 SPH 相关的 body（三层识别策略）
        
        1. SPH_SITE 标记的动态体（有力反馈耦合）
        2. _SPH_MESH_GEOM 标记的动态体（可能缺少 SPH_SITE）
        3. _SPH_STATIC_MESH_GEOM 标记的静态体（无锚点，仅边界碰撞）
        
        Returns:
            List[str]: body 名称列表
        """
        sph_bodies = set()
        
        try:
            body_names = self.model.get_body_names()
            site_dict = self.model.get_site_dict()
            geom_dict = self.model.get_geom_dict()
            
            # 第1层：SPH_SITE 标记的动态体（通过 BodyID 关联，兼容 Euler 下 site 名与 body 名不一致）
            for site_name, site_info in site_dict.items():
                if "SPH_SITE" in site_name and "SPH_MOCAP_SITE" not in site_name:
                    body_id = site_info.get("BodyID")
                    body_name = self.model.body_id2name(body_id) if body_id is not None else None
                    if body_name and body_name in body_names:
                        sph_bodies.add(body_name)
                        logger.debug(f"Identified SPH body via SPH_SITE: {body_name}")
            
            # 第2层：_SPH_MESH_GEOM 标记的动态体
            if geom_dict:
                for geom_name, geom_info in geom_dict.items():
                    if '_SPH_MESH_GEOM' in geom_name and '_SPH_STATIC_MESH_GEOM' not in geom_name:
                        body_name = geom_info.get('BodyName', '')
                        if body_name and body_name in body_names and body_name not in sph_bodies:
                            sph_bodies.add(body_name)
                            logger.info(f"Identified SPH body via _SPH_MESH_GEOM: {body_name}")
            
            # 第3层：_SPH_STATIC_MESH_GEOM 标记的静态体
            if geom_dict:
                for geom_name, geom_info in geom_dict.items():
                    if '_SPH_STATIC_MESH_GEOM' in geom_name:
                        body_name = geom_info.get('BodyName', '')
                        if body_name and body_name in body_names and body_name not in sph_bodies:
                            sph_bodies.add(body_name)
                            logger.info(f"Identified SPH static body via _SPH_STATIC_MESH_GEOM: {body_name}")
            
            result = sorted(list(sph_bodies))
            logger.info(f"Identified {len(result)} SPH bodies: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error identifying SPH bodies: {e}", exc_info=True)
            return []

    def _mujoco_body_id(self, body_name: str) -> int:
        """取 body 在 MuJoCo 模型里的真实 id（字典 ID 字段），不是 enumerate 下标。"""
        body_dict = self.model.get_body_dict() or {}
        info = body_dict.get(body_name) or {}
        if "ID" in info:
            return int(info["ID"])
        return int(self.model.body_name2id(body_name))
    
    def extract_sites_for_body(self, body_name: str) -> Dict[str, List[str]]:
        """
        为指定 body 提取所有 SPH_SITE 和 SPH_MOCAP_SITE

        SPH_SITE 通过 BodyID 关联到 body；
        SPH_MOCAP_SITE 挂在独立 mocap body 上，通过 SPH_SITE 的名称前缀
        （将 _SPH_SITE_ 替换为 _SPH_MOCAP_SITE_）匹配对应项。

        Args:
            body_name: MuJoCo body 名称

        Returns:
            Dict with keys:
                - 'sph_sites': List of SPH_SITE names (sorted by index)
                - 'mocap_sites': List of SPH_MOCAP_SITE names (sorted by index)
        """
        sph_sites = []
        mocap_sites = []

        try:
            site_dict = self.model.get_site_dict()
            body_id = self._mujoco_body_id(body_name)

            # 第1步：通过真实 BodyID 收集属于该 body 的 SPH_SITE
            for site_name, site_info in site_dict.items():
                site_body_id = site_info.get("BodyID")
                if site_body_id is None:
                    continue
                try:
                    if int(site_body_id) != body_id:
                        continue
                except (TypeError, ValueError):
                    continue
                if "SPH_SITE" in site_name and "SPH_MOCAP_SITE" not in site_name:
                    sph_sites.append(site_name)

            # 第2步：通过名称前缀匹配对应的 SPH_MOCAP_SITE
            # e.g. [id]_bodyjoint_SPH_SITE_000 -> [id]_bodyjoint_SPH_MOCAP_SITE_000
            for sph_site_name in sph_sites:
                mocap_site_name = sph_site_name.replace("_SPH_SITE_", "_SPH_MOCAP_SITE_")
                if mocap_site_name in site_dict:
                    mocap_sites.append(mocap_site_name)
            
            # 按索引排序
            def extract_index(name: str) -> int:
                """从 site 名称提取索引"""
                match = re.search(r'(\d+)$', name)
                return int(match.group(1)) if match else 999
            
            sph_sites.sort(key=extract_index)
            mocap_sites.sort(key=extract_index)
            
            logger.debug(f"Body '{body_name}': {len(sph_sites)} SPH_SITE, {len(mocap_sites)} MOCAP_SITE")
            
            return {
                'sph_sites': sph_sites,
                'mocap_sites': mocap_sites
            }
            
        except Exception as e:
            logger.error(f"Error extracting sites for body '{body_name}': {e}")
            return {'sph_sites': [], 'mocap_sites': []}
    
    def generate_connection_points(self, body_name: str, sph_sites: List[str], mocap_sites: List[str]) -> List[Dict]:
        """
        生成 connection_points 配置
        
        Args:
            body_name: MuJoCo body 名称
            sph_sites: SPH_SITE 名称列表（已排序）
            mocap_sites: SPH_MOCAP_SITE 名称列表（已排序）
        
        Returns:
            List[Dict]: connection_points 配置列表
        """
        connection_points = []
        
        # 确保两个列表长度一致
        min_len = min(len(sph_sites), len(mocap_sites))
        
        if len(sph_sites) != len(mocap_sites):
            logger.warning(f"Body '{body_name}': SPH_SITE count ({len(sph_sites)}) != MOCAP_SITE count ({len(mocap_sites)})")
        
        for i in range(min_len):
            site_name = sph_sites[i]
            mocap_site_name = mocap_sites[i]
            
            # 从 mocap_site_name 提取索引
            # "toys_usda_box_body_SPH_MOCAP_SITE_000" → 0
            match = re.search(r'SPH_MOCAP_SITE_(\d+)$', mocap_site_name)
            if match:
                index = int(match.group(1))
                # mocap body 名跟 SITE 一样用内部 ID 前缀，不能用描述性 body 名
                internal_prefix = site_name.split("_SPH_SITE_")[0]
                mocap_body_name = f"{internal_prefix}_SPH_MOCAP_{index:03d}"
            else:
                internal_prefix = resolve_internal_prefix(self.model, body_name)
                mocap_body_name = f"{internal_prefix}_SPH_MOCAP_{i:03d}"
            
            connection_points.append({
                "point_id": site_name,  # 使用实际 site 名称
                "site_name": site_name,
                "mocap_name": mocap_body_name  # 使用 mocap body 名称
            })
        
        return connection_points
    
    def generate_rigid_bodies_force_position(self) -> List[Dict]:
        """
        force_position 模式：按刚体生成配置。
        object_id = 内部 ID 前缀（给 SPH / OrcaLink）；
        mujoco_body = 描述性 body 名（给 Euler 查位姿）。
        仅包含带 SPH_SITE 的动力学耦合体。
        """
        rigid_bodies = []
        sph_bodies = self.identify_sph_bodies()
        for body_name in sph_bodies:
            sites_info = self.extract_sites_for_body(body_name)
            if not sites_info['sph_sites']:
                continue
            object_id = resolve_internal_prefix(self.model, body_name)
            rigid_bodies.append({
                "object_id": object_id,
                "mujoco_body": body_name,
                "coupling_mode": "force_position",
                "connection_points": [],
            })
            if object_id != body_name:
                logger.info(
                    f"Generated force_position body '{body_name}' "
                    f"(object_id='{object_id}')"
                )
            else:
                logger.info(f"Generated force_position body '{body_name}'")
        logger.info(f"Generated {len(rigid_bodies)} force_position rigid bodies")
        return rigid_bodies

    def generate_rigid_bodies(self) -> List[Dict]:
        """
        从 MuJoCo 模型生成完整的 rigid_bodies 配置
        
        Returns:
            List[Dict]: rigid_bodies 配置列表
        """
        rigid_bodies = []
        
        # 识别所有 SPH bodies
        sph_bodies = self.identify_sph_bodies()
        
        if not sph_bodies:
            logger.warning("No SPH bodies found in the model")
            return []
        
        # 为每个 body 生成配置
        for body_name in sph_bodies:
            # 提取 sites
            sites_info = self.extract_sites_for_body(body_name)
            sph_sites = sites_info['sph_sites']
            mocap_sites = sites_info['mocap_sites']
            
            object_id = resolve_internal_prefix(self.model, body_name)

            if not sph_sites:
                rigid_body = {
                    "object_id": object_id,
                    "mujoco_body": body_name,
                    "coupling_mode": "static_boundary",
                    "connection_points": []
                }
                rigid_bodies.append(rigid_body)
                logger.info(f"Generated static body config for '{body_name}' (no SPH_SITE)")
                continue
            
            # 生成 connection_points
            connection_points = self.generate_connection_points(body_name, sph_sites, mocap_sites)
            
            if not connection_points:
                rigid_body = {
                    "object_id": object_id,
                    "mujoco_body": body_name,
                    "coupling_mode": "static_boundary",
                    "connection_points": []
                }
                rigid_bodies.append(rigid_body)
                logger.info(f"Generated static body config for '{body_name}' (no connection points)")
                continue
            
            # 生成 rigid body 配置
            rigid_body = {
                "object_id": object_id,
                "mujoco_body": body_name,
                "coupling_mode": "multipoint_connect",
                "connection_points": connection_points,
                "spring_params": {
                    "stiffness": 5000.0,
                    "damping": 100.0
                }
            }
            
            rigid_bodies.append(rigid_body)
            logger.info(f"Generated config for body '{body_name}': {len(connection_points)} connection points")
        
        logger.info(f"Generated {len(rigid_bodies)} rigid body configurations")
        return rigid_bodies

