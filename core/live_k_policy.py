# -*- coding: utf-8 -*-
"""V5.8.4 线上 K 专项集合策略。

将 V5.8.2 研究阶段的 K-specific policy 接入实际三肖/20码预测：
1. 每个目标 K 独立学习信号权重与集合结构参数；
2. 仅使用当期之前的 OOS folds 拟合策略；
3. 以最近 N 期滚动样本外结果计算命中率，不用训练集成绩冒充回测；
4. 历史不足或策略拟合失败时由调用方回退旧模型。
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from .k_specific_policy import KPolicyConfig, fit_k_policy, predict_k_policy
from .pair_signal import transition_pair_scores


def predict_and_backtest_k(
    folds: Sequence[Mapping[str, Any]],
    current_prob: Mapping[str, Mapping[Any, float]],
    signal_names: Sequence[str],
    candidates: Sequence[Any],
    target_k: int,
    backtest_n: int = 30,
    config: KPolicyConfig | None = None,
) -> dict | None:
    """返回当前 top-K 预测和严格 walk-forward 命中序列；数据不足返回 None。

    folds 每一行的 prob 必须在其 actual 开出前生成。current_prob 则必须只由
    当前已知的完整历史生成，代表下一期预测特征。
    """
    cfg = config or KPolicyConfig()
    names = list(signal_names)
    cands = list(candidates)
    if not names or not cands or target_k <= 0 or target_k > len(cands):
        return None

    # 只保留目标标签有效且包含全部信号的折，避免缺失值被当成真实样本。
    usable = [
        dict(row) for row in folds
        if row.get("actual") in cands
        and isinstance(row.get("prob"), Mapping)
        and all(name in row["prob"] for name in names)
    ]
    if len(usable) < max(cfg.min_history + 10, 70):
        return None

    try:
        # 当前策略用全部已开奖 OOS 行训练；当前下一期特征不含真实标签。
        policy = fit_k_policy(usable, names, cands, target_k, cfg)
        current_pairs = transition_pair_scores([row["actual"] for row in usable], cands)
        current = predict_k_policy(
            {"prob": current_prob}, cands, policy, names, current_pairs
        )

        # 严格滚动回测：每个测试期只使用该期之前的 folds 训练策略。
        n_test = max(1, min(int(backtest_n), len(usable) - cfg.min_history))
        start = len(usable) - n_test
        hits: list[int] = []
        for idx in range(start, len(usable)):
            history = usable[:idx]
            if len(history) < cfg.min_history:
                continue
            local_policy = fit_k_policy(history, names, cands, target_k, cfg)
            pair_scores = transition_pair_scores([row["actual"] for row in history], cands)
            ranked = predict_k_policy(
                usable[idx], cands, local_policy, names, pair_scores
            )
            hits.append(int(usable[idx]["actual"] in set(ranked)))

        if not hits:
            return None
        return {
            "ranking": list(current),
            "hits": hits,
            "n": len(hits),
            "policy": {
                "target_k": policy.target_k,
                "horizon": policy.horizon,
                "weights": policy.weights,
                "lambda_pair": policy.lambda_pair,
                "lambda_diversity": policy.lambda_diversity,
                "gain": policy.gain,
                "structure_enabled": policy.structure_enabled,
            },
        }
    except (ValueError, KeyError, TypeError, ZeroDivisionError, OverflowError):
        # 研究策略不可用时，安全回退由调用方负责，不中断主预测流程。
        return None
