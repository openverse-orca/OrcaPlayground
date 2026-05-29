"""流体监测 CSV 数值格式：浮点列统一小数位数（与 OrcaLinkBridge C++ 输出一致）。"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Mapping

CSV_FLOAT_DECIMALS: int = 4


def fmt_f(x: Any) -> str:
    """将标量格式化为固定小数位；非有限浮点输出 ``nan`` / ``inf`` 文本。"""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if math.isnan(v):
        return "nan"
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    return f"{v:.{CSV_FLOAT_DECIMALS}f}"


def fmt_coupling_dict_row(row: Mapping[str, Any], fieldnames: Iterable[str]) -> Dict[str, str]:
    """steer_mjc_sph_couple.csv 单行：整型列保持整数，其余浮点四位小数。"""
    int_fields = {"coupling_cycle", "sph_cycle_matched", "valid_cycle_pair"}
    out: Dict[str, str] = {}
    for k in fieldnames:
        v = row.get(k, "")
        if v == "" or v is None:
            out[k] = ""
            continue
        if k in int_fields:
            try:
                out[k] = str(int(float(v)))
            except (TypeError, ValueError):
                out[k] = str(v)
            continue
        if isinstance(v, str) and v.strip().lower() == "nan":
            out[k] = "nan"
            continue
        try:
            fv = float(v)
            if math.isnan(fv):
                out[k] = "nan"
            else:
                out[k] = fmt_f(fv)
        except (TypeError, ValueError):
            out[k] = str(v)
    return out
