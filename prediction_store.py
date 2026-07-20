#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""预测历史持久化与命中核对层。

每次预测完成后, 由 GUI 将完整预测数据序列化保存到 prediction_history.json;
之后可回查, 并对照已加载的开奖记录核对命中(特码/平码/集合/生肖/六维度)。
"""
import os
import sys
import json
from dataclasses import asdict
from datetime import datetime

from dimensions import (wave_of, zodiac_of, tail_of,
                        big_small_of, odd_even_of, head_of, normalize_wave)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        os.system("chcp 65001 >nul 2>&1")

STORE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "prediction_history.json")
MAX_RUNS = 200  # 软上限: 超出丢弃最旧, 避免无界增长

# 六维度核对: 维度名 -> 实际开奖取值函数(作用于 Record)
_DIM_EXTRACTORS = {
    "波色": wave_of,
    "生肖": zodiac_of,
    "尾数": tail_of,
    "大小": big_small_of,
    "奇偶": odd_even_of,
    "头数": head_of,
}


def serialize_run(results: dict) -> dict:
    """将 _work() 的 done payload 序列化为可 JSON 化的预测记录。

    results 期望含: records, llm(可 None), groups, sp, dims, pools, wide, zp, zq, bt。
    不保存 report(体积大且查阅不需要)与完整 records(只留基准期元数据)。
    """
    records = results.get("records", [])
    last = records[-1] if records else None
    llm = results.get("llm")
    now = datetime.now()
    return {
        "run_id": now.strftime("%Y%m%d_%H%M%S"),
        "created_at": now.strftime("%Y-%m-%d %H:%M:%S"),
        "basis_expect": last.expect if last else "",
        "basis_open_time": last.open_time if last else "",
        "basis_special": last.special if last else 0,
        "history_count": len(records),
        "backtest_n": results.get("bt", 0),
        "llm_models": list(llm.inferred_models) if llm else [],
        "llm_reasoning": (llm.reasoning or "")[:500] if llm else "",
        "groups": [asdict(g) for g in results.get("groups", [])],
        "specials": [asdict(g) for g in results.get("sp", [])],
        "dims": [asdict(d) for d in results.get("dims", [])],
        "pools": [asdict(p) for p in results.get("pools", [])],
        "wide": asdict(results["wide"]) if results.get("wide") else {},
        "zodiac": asdict(results["zp"]) if results.get("zp") else {},
        "zodiac_quad": asdict(results["zq"]) if results.get("zq") else {},
        "zodiac_six": asdict(results["zs"]) if results.get("zs") else {},
    }


def load_runs() -> list:
    """读取全部历史预测记录(时间升序)。缺失或损坏返回空列表。"""
    if not os.path.exists(STORE_FILE):
        return []
    try:
        with open(STORE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _write_runs(runs: list) -> bool:
    try:
        with open(STORE_FILE, "w", encoding="utf-8") as f:
            json.dump(runs, f, ensure_ascii=False, indent=2)
        return True
    except OSError:
        return False


def save_run(run: dict) -> bool:
    """追加一条预测记录; 超过 MAX_RUNS 时丢弃最旧。失败不抛异常, 返回是否成功。"""
    runs = load_runs()
    runs.append(run)
    if len(runs) > MAX_RUNS:
        runs = runs[-MAX_RUNS:]
    return _write_runs(runs)


def delete_run(run_id: str) -> bool:
    """按 run_id 删除一条记录。"""
    runs = [r for r in load_runs() if r.get("run_id") != run_id]
    return _write_runs(runs)


def clear_runs() -> bool:
    """清空全部历史。"""
    return _write_runs([])


def _actual_next(records: list, basis_expect: str):
    """返回基准期(expect)之后的下一期实际开奖, 即预测目标。无则 None。

    records 已按 expect 升序排序, expects 为等长数字串, 字典序与数值序一致。
    """
    if not basis_expect:
        return None
    for r in records:
        if r.expect > basis_expect:
            return r
    return None


def verify_run(run: dict, records: list):
    """核对单条预测对实际开奖的命中情况。

    返回 dict(命中明细) 或 None(目标期尚未开奖)。
    """
    basis = run.get("basis_expect", "")
    actual = _actual_next(records, basis)
    if actual is None:
        return None
    actual_special = actual.special
    actual_regular = set(actual.regular)
    actual_zodiac = zodiac_of(actual)

    # 推荐完整组(groups[0])
    g0 = run["groups"][0] if run.get("groups") else {}
    full_special_hit = g0.get("special") == actual_special
    full_regular_hits = len(set(g0.get("regular", [])) & actual_regular)

    # 五组单颗特码
    specials = run.get("specials", [])
    special_hits = [s.get("special") == actual_special for s in specials]

    # 号码集合(小集合 + 20 颗大集合)
    pools = run.get("pools", [])
    pool_hits = [actual_special in set(p.get("numbers", [])) for p in pools]

    wide = run.get("wide", {})
    wide_hit = actual_special in set(wide.get("numbers", []))

    # 三生肖
    zod = run.get("zodiac", {})
    zodiac_hit = actual_zodiac in set(zod.get("zodiacs", []))

    # 四生肖
    zodq = run.get("zodiac_quad", {})
    zodiac_quad_hit = actual_zodiac in set(zodq.get("zodiacs", []))

    # 六生肖
    zods = run.get("zodiac_six", {})
    zodiac_six_hit = actual_zodiac in set(zods.get("zodiacs", []))

    # 六维度
    dim_hits = {}
    dim_actuals = {}
    for d in run.get("dims", []):
        name = d.get("name", "")
        fn = _DIM_EXTRACTORS.get(name)
        if fn is None:
            continue
        try:
            actual_val = fn(actual)
        except Exception:
            actual_val = None
        dim_actuals[name] = "" if actual_val is None else str(actual_val)
        if actual_val is None:
            dim_hits[name] = False
        else:
            # 波色维度经 normalize_wave 规范化后再比对(兼容历史英文 red/green/blue 预测);
            # 对其他维度 normalize_wave 为恒等映射, 不影响结果。
            pred_val = normalize_wave(str(d.get("value", "")))
            dim_hits[name] = pred_val == normalize_wave(str(actual_val))

    return {
        "actual_expect": actual.expect,
        "actual_special": actual_special,
        "actual_regular": list(actual.regular),
        "actual_zodiac": actual_zodiac,
        "full_special_hit": full_special_hit,
        "full_regular_hits": full_regular_hits,
        "special_hits": special_hits,
        "special_hit_any": any(special_hits),
        "pool_hits": pool_hits,
        "pool_top_hit": pool_hits[0] if pool_hits else False,
        "wide_hit": wide_hit,
        "zodiac_hit": zodiac_hit,
        "zodiac_quad_hit": zodiac_quad_hit,
        "zodiac_six_hit": zodiac_six_hit,
        "dim_hits": dim_hits,
        "dim_actuals": dim_actuals,
        "dim_hit_count": sum(1 for v in dim_hits.values() if v),
        "dim_total": len(dim_hits),
    }
