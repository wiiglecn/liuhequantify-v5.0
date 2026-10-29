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
from core.statistical_validation_v587 import (
    benjamini_hochberg, binomial_upper_tail, holm_adjust,
    moving_block_bootstrap_ci,
)
from tools.compare_live_k_policy import (
    load_clean_records, paired, stacking_topk, summary,
)

# 每个目标 K 使用独立策略和独立随机基线；不要横向比较不同 K 的原始命中率。
ZODIAC_TARGET_KS = (3, 4, 6)
RANDOM_BASELINES = {
    "three_zodiac": 3 / 12,
    "four_zodiac": 4 / 12,
    "six_zodiac": 6 / 12,
    "twenty_code": 20 / 49,
}


def parse_windows(value: str) -> list[int]:
    """解析并校验窗口列表，拒绝重复、非正数和不足 10 期的窗口。"""
    parts = value.split(",")
    if any(not part.strip() for part in parts):
        raise argparse.ArgumentTypeError("窗口列表不能包含空项")
    try:
        windows = [int(part.strip()) for part in parts]
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
                if not 1 <= month <= 12:
                    raise ValueError("月份超出范围")
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
    # 同一生肖信号折分别拟合 K=3/4/6，避免把三肖权重直接套用于四肖/六肖。
    zodiac_results = {
        k: predict_and_backtest_k(
            z_folds, current_z, zodiac.SIGNAL_NAMES, z_candidates, k, eval_n
        )
        for k in ZODIAC_TARGET_KS
    }
    w_result = predict_and_backtest_k(
        w_folds, current_w, wide.SIGNAL_NAMES, wide.ALL_NUMS, 20, eval_n
    )
    if any(result is None for result in zodiac_results.values()) or w_result is None:
        raise SystemExit("一个或多个 K 专项策略未能生成有效 OOS 结果")
    new_z = {k: result["hits"] for k, result in zodiac_results.items()}
    new_w = w_result["hits"]
    for k, hits in new_z.items():
        if len(hits) != eval_n:
            raise SystemExit(f"{k}肖 OOS 样本长度不一致：{len(hits)}，预期={eval_n}")
    if len(new_w) != eval_n:
        raise SystemExit(f"20码 OOS 样本长度不一致：{len(new_w)}，预期={eval_n}")

    old_z = {k: [] for k in ZODIAC_TARGET_KS}
    old_w = []
    first_eval_idx = len(z_folds) - eval_n
    for index in range(first_eval_idx, len(z_folds)):
        for k in (3, 4, 6):
            old_z[k].append(int(z_folds[index]["actual"] in set(
                stacking_topk(zodiac, z_folds, index, k, zodiac.SIGNAL_NAMES)
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
        windows_report[str(window)] = {
            "requested_n": window,
            "actual_n": n,
            "three_zodiac": evaluate_segment(new_z[3][-n:], old_z[3][-n:], RANDOM_BASELINES["three_zodiac"]),
            "four_zodiac": evaluate_segment(new_z[4][-n:], old_z[4][-n:], RANDOM_BASELINES["four_zodiac"]),
            "six_zodiac": evaluate_segment(new_z[6][-n:], old_z[6][-n:], RANDOM_BASELINES["six_zodiac"]),
            "twenty_code": evaluate_segment(new_w[-n:], old_w[-n:], RANDOM_BASELINES["twenty_code"]),
            "first_open_time": eval_dates[-n] if n else None,
            "last_open_time": eval_dates[-1] if n else None,
        }

    # V5.8.7：对每个目标/窗口做精确随机基线检验，并对整个比较族进行多重检验校正。
    # 窗口相互重叠，因此这些 p 值不应被解释为独立重复试验；Holm 校正可在任意依赖下控制 FWER。
    stat_rows = []
    for window_key, window_data in windows_report.items():
        n = window_data["actual_n"]
        for target_key, baseline in RANDOM_BASELINES.items():
            result = window_data[target_key]["k_specific"]
            raw_p = binomial_upper_tail(result["hits"], n, baseline)
            new_hits = (new_z[3] if target_key == "three_zodiac" else
                        new_z[4] if target_key == "four_zodiac" else
                        new_z[6] if target_key == "six_zodiac" else new_w)[-n:]
            old_hits = (old_z[3] if target_key == "three_zodiac" else
                        old_z[4] if target_key == "four_zodiac" else
                        old_z[6] if target_key == "six_zodiac" else old_w)[-n:]
            block_ci = moving_block_bootstrap_ci(
                new_hits, old_hits, block_size=min(5, n), repetitions=2000,
                seed=20261029 + n + len(stat_rows),
            )
            stat_rows.append({
                "window": window_key,
                "target": target_key,
                "n": n,
                "hits": result["hits"],
                "baseline": baseline,
                "raw_one_sided_binomial_p": raw_p,
                "block_bootstrap_paired_difference_95_ci_percentage_points": [
                    round(block_ci[0] * 100, 3), round(block_ci[1] * 100, 3)
                ],
            })
    raw_ps = [row["raw_one_sided_binomial_p"] for row in stat_rows]
    holm_ps = holm_adjust(raw_ps)
    bh_qs = benjamini_hochberg(raw_ps)
    for idx, row in enumerate(stat_rows):
        row["holm_adjusted_p"] = holm_ps[idx]
        row["benjamini_hochberg_q"] = bh_qs[idx]
    statistical_validation = {
        "version": "V5.8.7",
        "tests_count": len(stat_rows),
        "multiple_testing_family": "当前报告内所有目标 K x 所有窗口的比较合并为一个检验族",
        "methods": [
            "单侧精确二项上尾检验（零假设为对应 K 的理论随机命中率）",
            "Holm 家族错误率校正",
            "Benjamini-Hochberg FDR 校正（探索性，窗口重叠时需谨慎解释）",
            "新旧策略配对时间块 bootstrap 95% 区间（区块长度最多 5 期）",
        ],
        "results": stat_rows,
        "limitations": [
            "二项检验假设各期命中指标独立且零假设命中率正确；若开奖规则或标签口径不同需调整。",
            "重叠窗口高度相关，校正不能把它们变成独立证据。",
            "本报告的精确检验只评估已报告的命中数；尚未在模拟的完整开奖序列上重跑全部自适应搜索管线。",
            "通过统计门槛也只允许前瞻影子验证，不自动切换生产策略。",
        ],
    }

    # 默认最近 300 期可能只覆盖一个年份；报告会明确显示，不能将其冒充跨年验证。
    segments = {}
    for grouping in ("year", "quarter"):
        segments[grouping] = {}
        for key, positions in segment_indices(eval_dates, grouping).items():
            w_new = [new_w[i] for i in positions]
            w_old = [old_w[i] for i in positions]
            segments[grouping][key] = {
                "three_zodiac": evaluate_segment([new_z[3][i] for i in positions], [old_z[3][i] for i in positions], RANDOM_BASELINES["three_zodiac"]),
                "four_zodiac": evaluate_segment([new_z[4][i] for i in positions], [old_z[4][i] for i in positions], RANDOM_BASELINES["four_zodiac"]),
                "six_zodiac": evaluate_segment([new_z[6][i] for i in positions], [old_z[6][i] for i in positions], RANDOM_BASELINES["six_zodiac"]),
                "twenty_code": evaluate_segment(w_new, w_old, RANDOM_BASELINES["twenty_code"]),
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
            "random_baselines": RANDOM_BASELINES,
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
        "statistical_validation": statistical_validation,
        "decision": {
            "three_zodiac": assess_decision(windows_report, "three_zodiac"),
            "four_zodiac": assess_decision(windows_report, "four_zodiac"),
            "six_zodiac": assess_decision(windows_report, "six_zodiac"),
            "twenty_code": assess_decision(windows_report, "twenty_code"),
        },
        "segments": segments,
        "current_policy": {
            "three_zodiac": zodiac_results[3]["policy"],
            "four_zodiac": zodiac_results[4]["policy"],
            "six_zodiac": zodiac_results[6]["policy"],
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
                "four_zodiac": {
                    "old": v["four_zodiac"]["old_stacking"]["hit_rate_percent"],
                    "k_policy": v["four_zodiac"]["k_specific"]["hit_rate_percent"],
                    "n": v["actual_n"],
                },
                "six_zodiac": {
                    "old": v["six_zodiac"]["old_stacking"]["hit_rate_percent"],
                    "k_policy": v["six_zodiac"]["k_specific"]["hit_rate_percent"],
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
