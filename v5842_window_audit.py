#!/usr/bin/env python3
"""V5.8.4.2 rolling-window OOS audit.

Compares the legacy all-history Stacking window with bounded rolling windows.
Window selection is made only on the development OOS segment; the final
60 historical folds remain untouched until after selection.
"""
import json
import os
from data_fetcher import Record
import zodiac_ensemble as z
import wide_ensemble as w

WINDOWS = (60, 90, 120, 150, 240, 360, 10000)
FINAL_HOLDOUT = 60
DEV_MAX = 300


def load(path):
    raw = json.load(open(path, encoding="utf-8"))
    return [Record(expect=str(x["expect"]), open_time=str(x.get("openTime", "")),
                   regular=x["regular"], special=x["special"],
                   waves=x.get("waves", []), zodiacs=x.get("zodiacs", []))
            for x in raw]


def eval_zodiac(folds, start, end, window):
    hits = {3: 0, 4: 0, 6: 0}
    n = max(0, end - start)
    for i in range(start, end):
        rank = z._rank_stacking(folds, i, window=window)
        actual = folds[i]["actual"]
        for k in hits:
            hits[k] += int(actual in set(rank[:k]))
    return {"n": n, "rates": {str(k): (hits[k] / n if n else 0.0) for k in hits}}


def eval_wide(folds, start, end, window):
    hits = 0
    n = max(0, end - start)
    for i in range(start, end):
        rank = w._rank_stacking(folds, i, window=window)
        hits += int(folds[i]["actual"] in set(rank[:20]))
    return {"n": n, "rate": hits / n if n else 0.0}


def choose_zodiac(results):
    # Equal importance to 三肖/四肖/六肖; no final-holdout information used.
    return max(results, key=lambda x: sum(float(x["rates"][str(k)]) for k in (3, 4, 6)) / 3.0)


def choose_wide(results):
    return max(results, key=lambda x: float(x["rate"]))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="macau_history.json")
    ap.add_argument("--output-dir", default="reports/v5.8.4.2")
    a = ap.parse_args()
    records = load(a.input)

    zfolds = z.precompute_folds(records, max(1, len(records) - 10))
    wfolds = w.precompute_folds(records, max(1, len(records) - 10))

    def bounds(n):
        end = n - FINAL_HOLDOUT
        start = max(1, end - DEV_MAX)
        return start, end, end, n

    zds, zde, zhs, zhe = bounds(len(zfolds))
    wds, wde, whs, whe = bounds(len(wfolds))

    zdev = [{"window": win, **eval_zodiac(zfolds, zds, zde, win)} for win in WINDOWS]
    wdev = [{"window": win, **eval_wide(wfolds, wds, wde, win)} for win in WINDOWS]
    zbest = choose_zodiac(zdev)["window"]
    wbest = choose_wide(wdev)["window"]

    zhold = eval_zodiac(zfolds, zhs, zhe, zbest)
    whold = eval_wide(wfolds, whs, whe, wbest)

    result = {
        "version": "V5.8.4.2",
        "records": len(records),
        "final_holdout": FINAL_HOLDOUT,
        "development_oos_max": DEV_MAX,
        "zodiac": {"development": zdev, "selected_window": zbest,
                   "final_holdout": zhold, "holdout_range": [zhs, zhe]},
        "wide": {"development": wdev, "selected_window": wbest,
                 "final_holdout": whold, "holdout_range": [whs, whe]},
        "protocol": "development_OOS_select_then_final_60_holdout",
    }
    os.makedirs(a.output_dir, exist_ok=True)
    with open(os.path.join(a.output_dir, "window_audit.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with open(os.path.join(a.output_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("# V5.8.4.2 Rolling Window OOS Audit\n\n")
        f.write("Protocol: development OOS selects the window; final 60 folds are isolated.\n\n")
        f.write("## Zodiac\n\n| Window | N | 三肖 | 四肖 | 六肖 |\n|---:|---:|---:|---:|---:|\n")
        for x in zdev:
            f.write(f'| {x["window"]} | {x["n"]} | {x["rates"]["3"]:.4f} | {x["rates"]["4"]:.4f} | {x["rates"]["6"]:.4f} |\n')
        f.write(f'\nSelected window: **{zbest}**. Final-60: 三肖={zhold["rates"]["3"]:.4f}, 四肖={zhold["rates"]["4"]:.4f}, 六肖={zhold["rates"]["6"]:.4f}.\n')
        f.write("\n## Wide\n\n| Window | N | 20码 |\n|---:|---:|---:|\n")
        for x in wdev:
            f.write(f'| {x["window"]} | {x["n"]} | {x["rate"]:.4f} |\n')
        f.write(f'\nSelected window: **{wbest}**. Final-60: 20码={whold["rate"]:.4f}.\n')


if __name__ == "__main__":
    main()

# CI trigger: execute the real 999-record OOS window audit on dev.
