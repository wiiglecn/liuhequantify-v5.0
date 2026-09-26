# -*- coding: utf-8 -*-
"""新六维模型逐期命中复盘 (walk-forward, 与 diag_dims.py 同口径)。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["PYTHONUTF8"] = "1"
from collections import Counter
from data_fetcher import load_cache
from dimensions import (predict_dimensions, DIMENSIONS, build_zodiac_map,
                        comb_prior, _dim_predict_value, DIM_CONFIG)

records = load_cache()
N = len(records)
WINDOW = 21  # 与 diag_dims 可核对 run 数一致

DIM_ORDER = [n for n, _ in DIMENSIONS]
print("  basis   target  " + "  ".join(f"{n}" for n in DIM_ORDER) + "   hit")
print("-" * 100)

hit = Counter(); total = Counter()
for i in range(WINDOW):
    split = N - WINDOW + i
    train = records[:split]
    actual = records[split]
    zmap = build_zodiac_map(train)
    cells = []
    cnt = 0
    for name, ex in DIMENSIONS:
        prior = comb_prior(name, zmap)
        if not prior:
            cells.append("?")
            continue
        pred = _dim_predict_value(train, ex, prior, DIM_CONFIG.get(name))
        act = ex(actual)
        ok = (str(pred) == str(act))
        total[name] += 1
        if ok:
            hit[name] += 1
            cnt += 1
            cells.append(f"OK {pred}={act}")
        else:
            cells.append(f"XX {pred}!={act}")
    print(f"  {train[-1].expect}  {actual.expect}  " + "  ".join(f"{c}" for c in cells) + f"   {cnt}/6")

print("-" * 100)
print(f"\n各维度命中率 (近 {WINDOW} 期, 对照精确组合基线):\n")
BASE = {"波色": "17/49≈34.7%", "生肖": "5/49≈10.2%", "尾数": "5/49≈10.2%",
        "大小": "25/49≈51.0%", "奇偶": "25/49≈51.0%", "头数": "10/49≈20.4%"}
for name in DIM_ORDER:
    t = total[name]; h = hit[name]
    rate = f"{h}/{t} = {h/t:.0%}" if t else "n/a"
    print(f"  {name}: {rate}   (基线 {BASE[name]})")
