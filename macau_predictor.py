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

    print("[1/6] 加载历史数据...")
    records = load_history(years, refresh=args.refresh)
    if len(records) < 30:
        print("  ! 数据不足, 请检查网络或强制刷新(--refresh)。当前期数:", len(records))
        return
    print(f"  ✓ 共 {len(records)} 期, 时间范围 {records[0].open_time} ~ {records[-1].open_time}")
    print(f"  ✓ 最近一期: 期号 {records[-1].expect} 平码 {records[-1].regular} 特码 {records[-1].special}")
    print()

    print("[2/6] 统计分析与模型反推...")
    report = build_report(records)
    print("  反推理候选数学模型(拟合度 0-1, 越高越可能解释该序列):")
    for name, score, detail in report["models"]:
        print(f"    - {name}: {score:.3f}  ({detail})")
    print()

    print("[3/6] 大模型反推理(可选)...")
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

    print("[4/6] 三组预测与回测...")
    groups = predict_all(records, llm_result=llm_result, backtest_n=args.backtest)
    for i, g in enumerate(groups):
        star = "  ★推荐" if i == 0 else ""
        print(f"  {g.name}组{star}: 平码 {g.regular}  特码 {g.special}")
        print(f"      策略: {g.strategy}")
        print(f"      回测: 特码命中率={g.backtest_special_hit:.1%}  平均平码命中={g.backtest_regular_hits:.2f}/6")
    print()

    print("[5/6] 五组特码预测与回测...")
    sp_groups = predict_special_groups(records, llm_result=llm_result, backtest_n=args.backtest)
    for i, g in enumerate(sp_groups):
        star = "  ★推荐" if i == 0 else ""
        print(f"  特码{g.name}组{star}: 特码 {g.special}")
        print(f"      策略: {g.strategy}")
        print(f"      回测: 特码命中率={g.backtest_hit:.1%}")
    print()

    print("[6/6] 完成")
    print("-" * 70)
    print(DISCLAIMER)


if __name__ == "__main__":
    main()
