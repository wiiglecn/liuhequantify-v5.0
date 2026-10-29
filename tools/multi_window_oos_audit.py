#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V5.8.5：三肖/20码多窗口严格滚动样本外稳定性审计。"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import zodiac_ensemble as zodiac
import wide_ensemble as wide
from data_fetcher import Record
from dimensions import build_zodiac_map
from core.live_k_policy import predict_and_backtest_k
from tools.compare_live_k_policy import (
    load_clean_records, paired, stacking_topk, summary,
)


def parse_windows(value: str) -> list[int]:
    """解析并校验窗口列表，拒绝重复、非正数和不足 10 期的窗口。"""
    try:
        windows = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("窗口必须是逗号分隔的整数") from exc
    if not windows or any(n < 10 for n in windows):
        raise argparse.ArgumentTypeError("至少指定一个窗口，且每个窗口不少于 10 期")
    if len(set(windows)) != len(windows):
        raise argparse.ArgumentTypeError("窗口不能重复")
    return sorted(windows)


def segment_indices(dates: list[str], grouping: str) -> dict[str, list[int]]:
    """按实际开奖日期划分年度或季度，返回各分段在 OOS 序列中的位置。"""
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, date in enumerate(dates):
        if len(date) < 7:
            key = "unknown"
        elif grouping == "year":
            key = date[:4]
        else:
            try:
                month = int(date[5:7])
                key = f"{date[:4]}-Q{(month - 1) // 3 + 1}"
            except ValueError:
                key = "unknown"
        groups[key].append(idx)
    return dict(sorted(groups.items()))


def evaluate_segment(new_hits: list[int], old_hits: list[int], baseline: float) -> dict:
    """输出模型、随机基线和新旧策略配对指标。"""
    return {
        "n": len(new_hits),
        "old_stacking": summary(old_hits, baseline),
        "k_specific": summary(new_hits, baseline),
        "paired_difference": paired(new_hits, old_hits),
    }


def assess_decision(windows_report: dict, target_key: str) -> dict:
    """使用预先声明的保守门槛给出研究决策，不直接自动切换生产策略。"""
    required = ("120", "180", "300")
    missing = [key for key in required if key not in windows_report or
               windows_report[key]["actual_n"] < int(key)]
    if missing:
        return {
            "decision": "insufficient_evidence",
            "reason": f"窗口数据不足：{', '.join(missing)}",
            "production_switch_allowed": False,
        }
    rows = [windows_report[key][target_key] for key in required]
    old_rates = [row["old_stacking"]["hit_rate"] for row in rows]
    new_rates = [row["k_specific"]["hit_rate"] for row in rows]
    baselines = [row["k_specific"]["baseline"] for row in rows]
    ci = rows[-1]["paired_difference"].get("paired_bootstrap_95_ci_percentage_points")
    stable_lift = all(new >= old and new >= base for new, old, base in zip(new_rates, old_rates, baselines))
    positive_300_ci = bool(ci and ci[0] > 0)
    eligible = stable_lift and positive_300_ci
    return {
        "decision": "candidate_for_prospective_validation" if eligible else "retain_old_or_shadow_mode",
        "criteria": {
            "k_policy_not_worse_than_old_and_at_or_above_random_baseline_in_all_windows": stable_lift,
            "300_draw_paired_bootstrap_95_ci_lower_bound_above_zero_percentage_points": positive_300_ci,
        },
        "window_hit_rates": {
            key: {
                "old_stacking_percent": windows_report[key][target_key]["old_stacking"]["hit_rate_percent"],
                "k_specific_percent": windows_report[key][target_key]["k_specific"]["hit_rate_percent"],
                "random_baseline_percent": windows_report[key][target_key]["k_specific"]["baseline_percent"],
            }
            for key in required
        },
        "production_switch_allowed": False,
        "reason": "即使满足历史门槛，也必须先冻结代码与评估口径并完成前瞻验证。",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="macau_history.json")
    parser.add_argument("--output", default="reports/v585_multi_window_oos.json")
    parser.add_argument("--windows", type=parse_windows, default=parse_windows("120,180,300"))
    parser.add_argument(
        "--full-history", action="store_true",
        help="计算可用的全历史滚动 OOS 序列；耗时明显高于默认最近 300 期审计",
    )
    args = parser.parse_args()

    records, audit = load_clean_records(args.input)
    if len(records) < 150:
        raise SystemExit(f"历史样本不足：{len(records)}")
    max_window = max(args.windows)
    # 为 300 期目标窗口保留额外训练折；全历史模式用于年度分组研究。
    n_folds = len(records) - 10 if args.full_history else min(len(records) - 10, max(200, max_window + 150))
    z_folds = zodiac.precompute_folds(records, n_folds)
    w_folds = wide.precompute_folds(records, n_folds)

    zmap = build_zodiac_map(records)
    current_z, _ = zodiac._compute_fold_probs(records, zmap)
    current_w = wide._compute_fold_probs(records)
    z_candidates = sorted(set(zmap.values()))
    if len(z_folds) != len(w_folds):
        raise SystemExit("三肖与20码预计算折数量不一致，拒绝比较")

    # 每个目标只执行一次最长 OOS 回测，较短窗口取同一序列后缀，保证可比。
    eval_n = (len(z_folds) - 60) if args.full_history else min(max_window, len(z_folds) - 60)
    if eval_n < max_window and not args.full_history:
        print(f"提示：可用 OOS 样本只有 {eval_n} 期，部分目标窗口将缩短。")
    z_result = predict_and_backtest_k(
        z_folds, current_z, zodiac.SIGNAL_NAMES, z_candidates, 3, eval_n
    )
    w_result = predict_and_backtest_k(
        w_folds, current_w, wide.SIGNAL_NAMES, wide.ALL_NUMS, 20, eval_n
    )
    if z_result is None or w_result is None:
        raise SystemExit("K 专项策略未能生成有效 OOS 结果")
    new_z, new_w = z_result["hits"], w_result["hits"]
    if len(new_z) != len(new_w) or len(new_z) != eval_n:
        raise SystemExit(f"OOS 样本长度不一致：三肖={len(new_z)}，20码={len(new_w)}，预期={eval_n}")

    old_z, old_w = [], []
    first_eval_idx = len(z_folds) - eval_n
    for index in range(first_eval_idx, len(z_folds)):
        old_z.append(int(z_folds[index]["actual"] in set(
            stacking_topk(zodiac, z_folds, index, 3, zodiac.SIGNAL_NAMES)
        )))
        old_w.append(int(w_folds[index]["actual"] in set(
            stacking_topk(wide, w_folds, index, 20, wide.SIGNAL_NAMES)
        )))

    # 预计算折与原始开奖记录按尾部索引对齐，分段时使用实际开奖日期。
    first_record_idx = len(records) - n_folds + first_eval_idx
    eval_dates = [r.open_time for r in records[first_record_idx:first_record_idx + eval_n]]
    if len(eval_dates) != eval_n:
        raise SystemExit("OOS 日期与命中序列长度不一致")

    windows_report = {}
    for window in args.windows:
        n = min(window, eval_n)
        new_z_w, old_z_w = new_z[-n:], old_z[-n:]
        new_w_w, old_w_w = new_w[-n:], old_w[-n:]
        windows_report[str(window)] = {
            "requested_n": window,
            "actual_n": n,
            "three_zodiac": evaluate_segment(new_z_w, old_z_w, 3 / 12),
            "twenty_code": evaluate_segment(new_w_w, old_w_w, 20 / 49),
            "first_open_time": eval_dates[-n] if n else None,
            "last_open_time": eval_dates[-1] if n else None,
        }

    # 默认最近 300 期可能只覆盖一个年份；报告会明确显示，不能将其冒充跨年验证。
    segments = {}
    for grouping in ("year", "quarter"):
        segments[grouping] = {}
        for key, positions in segment_indices(eval_dates, grouping).items():
            z_new = [new_z[i] for i in positions]
            z_old = [old_z[i] for i in positions]
            w_new = [new_w[i] for i in positions]
            w_old = [old_w[i] for i in positions]
            segments[grouping][key] = {
                "three_zodiac": evaluate_segment(z_new, z_old, 3 / 12),
                "twenty_code": evaluate_segment(w_new, w_old, 20 / 49),
                "first_open_time": eval_dates[positions[0]],
                "last_open_time": eval_dates[positions[-1]],
            }

    report = {
        "version": "V5.8.5_MULTI_WINDOW_OOS_AUDIT",
        "history": audit,
        "evaluation": {
            "mode": "full_history" if args.full_history else "recent_windows",
            "requested_windows": args.windows,
            "oos_n": eval_n,
            "first_open_time": eval_dates[0] if eval_dates else None,
            "last_open_time": eval_dates[-1] if eval_dates else None,
            "strict_walk_forward": True,
            "random_baselines": {"three_zodiac": 3 / 12, "twenty_code": 20 / 49},
            "holdout_status": (
                "历史最近 120 期不是未触碰的独立留出集；最近 30 期已在前一轮审计中查看。"
                "请冻结当前实现与评估口径，未来新增开奖用于真正的前瞻验证。"
            ),
            "limitations": [
                "彩票开奖结果具有随机性，历史命中率不保证未来优势。",
                "窗口之间高度重叠，不能把多个窗口视为独立重复实验。",
                "按年结果只有覆盖到的年份才会出现；默认模式若只覆盖一年，不代表跨年稳定。",
                "Wilson 区间和配对 bootstrap 区间均为描述性统计，不等于因果或盈利证明。",
            ],
        },
        "windows": windows_report,
        "segments": segments,
        "current_policy": {
            "three_zodiac": z_result["policy"],
            "twenty_code": w_result["policy"],
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    compact = {
        "output": str(output),
        "history": audit,
        "evaluation": report["evaluation"],
        "windows": {
            k: {
                "three_zodiac": {
                    "old": v["three_zodiac"]["old_stacking"]["hit_rate_percent"],
                    "k_policy": v["three_zodiac"]["k_specific"]["hit_rate_percent"],
                    "n": v["actual_n"],
                },
                "twenty_code": {
                    "old": v["twenty_code"]["old_stacking"]["hit_rate_percent"],
                    "k_policy": v["twenty_code"]["k_specific"]["hit_rate_percent"],
                    "n": v["actual_n"],
                },
            }
            for k, v in windows_report.items()
        },
        "year_segments": list(segments["year"]),
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
