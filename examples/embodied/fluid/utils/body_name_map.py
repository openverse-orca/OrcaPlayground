"""
流体专用：内部 ID 前缀 ↔ 描述性 body 名。

只给 OrcaPlayground 流体样例用，不要给布料 / XPBD / OrcaLink 引用。
现场 Group 场景里，Euler `_body_dict` 的键是描述性 body 名，
SPH SITE/mesh 和 OrcaLink object_id 常用内部 ID 前缀；两边对不上会 KeyError。
"""

from typing import Any, Dict

import logging

logger = logging.getLogger(__name__)

_SPH_SITE_MARK = "_SPH_SITE_"
_SPH_MOCAP_SITE_MARK = "_SPH_MOCAP_SITE_"
_SPH_MESH_GEOM_SUFFIX = "_SPH_MESH_GEOM"
_SPH_STATIC_MESH_GEOM_SUFFIX = "_SPH_STATIC_MESH_GEOM"


def _site_dict(model: Any) -> Dict[str, Any]:
    if model is None or not hasattr(model, "get_site_dict"):
        return {}
    return model.get_site_dict() or {}


def _geom_dict(model: Any) -> Dict[str, Any]:
    if model is None or not hasattr(model, "get_geom_dict"):
        return {}
    return model.get_geom_dict() or {}


def _body_names(model: Any) -> set:
    if model is None or not hasattr(model, "get_body_names"):
        return set()
    return set(model.get_body_names() or [])


def _mujoco_id_of_body(model: Any, body_name: str):
    """描述性 body 名 → MuJoCo 原生 body id。优先读字典里的 ID 字段。"""
    if hasattr(model, "get_body_dict"):
        info = (model.get_body_dict() or {}).get(body_name) or {}
        if "ID" in info:
            try:
                return int(info["ID"])
            except (TypeError, ValueError):
                pass
    try:
        return int(model.body_name2id(body_name))
    except Exception:
        return None


def _body_name_from_mujoco_id(model: Any, mujoco_id: int) -> str:
    """用 MuJoCo 原生 body id 反查描述性 body 名。

    先走 body_id2name；对不上时再扫 body 字典里的 ID 字段。
    后一条是为了避开 enumerate 下标和真实 id 不一致的情况。
    """
    try:
        name = model.body_id2name(int(mujoco_id))
        if name:
            return str(name)
    except Exception:
        pass
    if not hasattr(model, "get_body_dict"):
        return ""
    body_dict = model.get_body_dict() or {}
    for name, info in body_dict.items():
        try:
            if int(info.get("ID", -1)) == int(mujoco_id):
                return str(name)
        except (TypeError, ValueError):
            continue
    return ""


def resolve_internal_prefix(model: Any, body_name: str) -> str:
    """描述性 body 名 → SPH / OrcaLink 用的内部 ID 前缀。

    同一刚体上 SITE 名常是 `[id]_bodyjoint_SPH_SITE_000`，
    而 Euler body 名是 `Group_Interactive_..._bodyjoint`。
    用 SITE 的 BodyID 对上该 body 后，取 SITE 名在 `_SPH_SITE_` 之前的一段。
    没有 SITE 时，再从 SPH mesh geom 名去后缀。
    都找不到则原样返回 body_name（短链 XML 里两套本来就是同一串）。
    """
    if not body_name:
        return body_name
    target_body_id = _mujoco_id_of_body(model, body_name)

    if target_body_id is not None:
        for site_name, site_data in _site_dict(model).items():
            if _SPH_SITE_MARK not in site_name:
                continue
            if _SPH_MOCAP_SITE_MARK in site_name:
                continue
            site_body_id = site_data.get("BodyID")
            if site_body_id is None:
                continue
            try:
                if int(site_body_id) != target_body_id:
                    continue
            except (TypeError, ValueError):
                continue
            return site_name.split(_SPH_SITE_MARK)[0]

    for geom_name, geom_info in _geom_dict(model).items():
        if geom_info.get("BodyName") != body_name:
            continue
        if geom_name.endswith(_SPH_STATIC_MESH_GEOM_SUFFIX):
            return geom_name[: -len(_SPH_STATIC_MESH_GEOM_SUFFIX)]
        if geom_name.endswith(_SPH_MESH_GEOM_SUFFIX):
            return geom_name[: -len(_SPH_MESH_GEOM_SUFFIX)]
    return body_name


def resolve_mujoco_body_from_object_id(model: Any, object_id: str) -> str:
    """SPH / OrcaLink 的 object_id → Euler 施力用的描述性 body 名。

    1. object_id 本身已在 body 字典里：短链同名，直接用。
    2. 否则按内部 ID 前缀找挂在该刚体上的 SPH_SITE，再用 BodyID 反查 body 名。
    3. 再不行就从 SPH mesh geom 名匹配。
    都失败则返回原 object_id，由调用方按原逻辑报错。
    """
    if not object_id:
        return object_id
    names = _body_names(model)
    if object_id in names:
        return object_id

    prefix = object_id
    for site_name, site_data in _site_dict(model).items():
        if _SPH_SITE_MARK not in site_name:
            continue
        if _SPH_MOCAP_SITE_MARK in site_name:
            continue
        if site_name.split(_SPH_SITE_MARK)[0] != prefix:
            continue
        site_body_id = site_data.get("BodyID")
        if site_body_id is None:
            continue
        body_name = _body_name_from_mujoco_id(model, int(site_body_id))
        if body_name and body_name in names:
            return body_name

    for geom_name, geom_info in _geom_dict(model).items():
        geom_prefix = ""
        if geom_name.endswith(_SPH_STATIC_MESH_GEOM_SUFFIX):
            geom_prefix = geom_name[: -len(_SPH_STATIC_MESH_GEOM_SUFFIX)]
        elif geom_name.endswith(_SPH_MESH_GEOM_SUFFIX):
            geom_prefix = geom_name[: -len(_SPH_MESH_GEOM_SUFFIX)]
        if geom_prefix != prefix:
            continue
        body_name = str(geom_info.get("BodyName") or "")
        if body_name and body_name in names:
            return body_name
    return object_id


class FluidBodyNameMap:
    """一次建模、反复查询的流体名字映射缓存。

    施力每宏步会来三个 object_id，缓存避免反复扫 site/geom 字典。
    """

    def __init__(self, model: Any):
        self._model = model
        self._object_id_to_body: Dict[str, str] = {}

    def mujoco_body_name(self, object_id: str) -> str:
        """查 SPH object_id 对应的 MuJoCo body 名；结果写入缓存。"""
        cached = self._object_id_to_body.get(object_id)
        if cached is not None:
            return cached
        body_name = resolve_mujoco_body_from_object_id(self._model, object_id)
        self._object_id_to_body[object_id] = body_name
        if body_name != object_id:
            logger.info(
                "流体名字映射: object_id '%s' → body '%s'",
                object_id,
                body_name,
            )
        return body_name
