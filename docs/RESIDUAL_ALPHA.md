# V5.8.3 Residual Alpha / 信号残差发现

## 目标
V5.8.2 已经把“不同 K 使用不同策略”建立起来，但仍缺少一个独立问题：基线已经解释掉的部分，某个信号是否还能提供增量信息？

V5.8.3 不重新训练整个预测器，而是把 V5.8.2 K-specific Set Prediction 作为冻结基线，在 OOS 空间中研究候选信号的 residual alpha。

## 核心流程
1. 每个 OOS fold 先按历史数据拟合 V5.8.2 K-specific baseline。
2. 记录 baseline probability 与每个原始 signal probability。
3. 对每个候选 signal 搜索 residual lambda。
4. 目标是 paired delta LogLoss = baseline LogLoss - residual-blend LogLoss。
5. 候选选择只使用更早的 OOS rows。
6. 使用 sign-test、Bootstrap CI、BH 与 Bonferroni 作为统计审计。
7. 稳定性要求前后两个时间段都不出现负向平均增量。
8. 最终 60 期完全冻结，只用于最终评估。

## 接受门槛
默认：
- mean delta LogLoss >= 0.005
- Bootstrap CI lower > 0
- BH q <= 0.10
- 前后两段平均增量 >= 0
- residual lambda > 0

若无候选通过，系统明确返回 NONE，不强行加入 residual layer。

## 防泄漏边界
- baseline policy 的拟合只看 target 之前的历史。
- residual candidate 的选择只看更早的 OOS rows。
- V5.8.2 pair structure 在 V5.8.3 默认关闭，避免把结构项与 residual alpha 混在同一层审计。
- final holdout 不参与任何 signal/lambda/policy 选择。

## 输出
- residual_alpha_registry.json
- report.md
- 每个 K 的候选信号、lambda、平均 delta LogLoss、p-value、BH q-value、Bootstrap CI、稳定性、是否接受
- final holdout frozen diagnostics

这里的“alpha”只表示历史 OOS 中相对基线的增量信息量，不代表对未来开奖结果的保证。
