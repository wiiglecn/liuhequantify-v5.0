#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""澳门六合彩开奖记录分析与预测软件 — 主入口"""
import os
import sys
import argparse
from datetime import datetime

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        os.system("chcp 65001 >nul 2>&1")

from data_fetcher import load_history
from analysis import build_report, summarize_for_llm
from llm_reasoner import load_config, reason
from predictor import predict_all, predict_special_groups
from dimensions import predict_dimensions
from special_pool import predict_special_pools, predict_wide_pool

DISCLAIMER = (
    "科学提示: 彩票开奖本质为独立随机事件, 任何模型的真实命中率理论上均接近随机概率\n"
    "(特码命中概率约 1/49≈2%, 单个平码命中约 6/49)。本软件的命中率评估基于历史回测,\n"
    "不代表未来表现, 结果仅供数学/统计研究与方法验证, 不构成任何投注建议。"
)


def box(title: str) -> str:
    width = 70
    top = "╔" + "═" * width + "╗"
    mid = "║" + title.center(width) + "║"
    bot = "╚" + "═" * width + "╝"
    return "\n".join([top, mid, bot])


def main():
    parser = argparse.ArgumentParser(description="澳门六合彩开奖记录分析与预测")
    parser.add_argument("--refresh", action="store_true", help="强制重新抓取数据")
    parser.add_argument("--years", default="2024,2025,2026", help="抓取年份(逗号分隔)")
    parser.add_argument("--backtest", type=int, default=30, help="回测期数")
    args = parser.parse_args()

    years = [int(y.strip()) for y in args.years.split(",") if y.strip()]

    print(box("澳门六合彩开奖记录分析与预测软件"))
    print("运行时间:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print()

    print("[1/9] 加载历史数据...")
    records = load_history(years, refresh=args.refresh)
    if len(records) < 30:
        print("  ! 数据不足, 请检查网络或强制刷新(--refresh)。当前期数:", len(records))
        return
    print(f"  ✓ 共 {len(records)} 期, 时间范围 {records[0].open_time} ~ {records[-1].open_time}")
    print(f"  ✓ 最近一期: 期号 {records[-1].expect} 平码 {records[-1].regular} 特码 {records[-1].special}")
    print()

    print("[2/9] 统计分析与模型反推...")
    report = build_report(records)
    print("  反推理候选数学模型(拟合度 0-1, 越高越可能解释该序列):")
    for name, score, detail in report["models"]:
        print(f"    - {name}: {score:.3f}  ({detail})")
    print()

    print("[3/9] 大模型反推理(可选)...")
    cfg = load_config()
    llm_result = None
    if cfg:
        try:
            recent = [r.special for r in records[-10:]]
            summary = summarize_for_llm(report, recent_n=10)
            llm_result = reason(summary, recent, cfg)
            print("  ✓ 大模型反推理完成, 模型清单:",
                  ", ".join(llm_result.inferred_models) or "(未解析)")
        except Exception as e:
            print(f"  ! 大模型调用失败, 降级为纯统计: {e}")
    else:
        print("  ! 未配置 LLM(编辑 config.ini 填入 api_key), 跳过大模型, 仅用统计层。")
    print()

    print("[4/9] 三组预测与回测...")
    groups = predict_all(records, llm_result=llm_result, backtest_n=args.backtest)
    for i, g in enumerate(groups):
        star = "  ★推荐" if i == 0 else ""
        print(f"  {g.name}组{star}: 平码 {g.regular}  特码 {g.special}")
        print(f"      策略: {g.strategy}")
        print(f"      回测: 特码命中率={g.backtest_special_hit:.1%}  平均平码命中={g.backtest_regular_hits:.2f}/6")
    print()

    print("[5/9] 五组特码预测与回测...")
    sp_groups = predict_special_groups(records, llm_result=llm_result, backtest_n=args.backtest)
    print(f"  (随机基线 = 1/49 ≈ {1/49:.1%})")
    for i, g in enumerate(sp_groups):
        star = "  ★推荐" if i == 0 else ""
        print(f"  特码{g.name}组{star}: 特码 {g.special}")
        print(f"      策略: {g.strategy}")
        print(f"      回测: 命中率={g.backtest_hit:.1%}  lift={g.lift:+.1%}  标准误={g.std_error:.1%}  稳定性={g.stability:.2f}")
    print()

    print("[6/9] 六维度特码属性预测与回测...")
    dims = predict_dimensions(records, llm_result=llm_result, backtest_n=args.backtest)
    print(f"  {'维度':<6}{'预测值':<8}{'准确率':>8}{'随机基线':>10}{'lift':>8}{'标准误':>8}{'稳定性':>8}")
    for d in dims:
        print(f"  {d.name:<6}{d.value:<8}{d.accuracy:>8.1%}{d.baseline:>10.1%}"
              f"{d.lift:>+8.1%}{d.std_error:>8.1%}{d.stability:>8.2f}")
    print("  (lift>0 表示优于随机基线; 稳定性越小越稳定)")
    print()

    print("[7/9] 特码号码集合预测与回测(每集合8~10颗)...")
    pools = predict_special_pools(records, llm_result=llm_result, backtest_n=args.backtest)
    for i, p in enumerate(pools):
        star = "  ★推荐" if i == 0 else ""
        nums = ",".join(f"{n:02d}" for n in p.numbers)
        print(f"  集合{p.name}组{star} ({len(p.numbers)}颗): {nums}")
        print(f"      策略: {p.strategy}")
        print(f"      回测: 命中率={p.hit_rate:.1%}  基线={p.baseline:.1%}  lift={p.lift:+.1%}"
              f"  标准误={p.std_error:.1%}  稳定性={p.stability:.2f}")
    print("  (命中=真实特码落在集合内; 随机基线=集合大小/49)")
    print()

    print("[8/9] 特码大集合预测(20颗)与回测...")
    wide = predict_wide_pool(records, llm_result=llm_result, backtest_n=args.backtest)
    nums = ",".join(f"{n:02d}" for n in wide.numbers)
    print(f"  ★大集合 ({len(wide.numbers)}颗): {nums}")
    print(f"      策略: {wide.strategy}")
    print(f"      回测: 命中率={wide.hit_rate:.1%}  基线={wide.baseline:.1%}  lift={wide.lift:+.1%}"
          f"  标准误={wide.std_error:.1%}  稳定性={wide.stability:.2f}")
    print("  (命中=真实特码落在 20 颗内; 随机基线=20/49≈40.8%)")
    print()

    print("[9/9] 完成")
    print("-" * 70)
    print(DISCLAIMER)


if __name__ == "__main__":
    main()
