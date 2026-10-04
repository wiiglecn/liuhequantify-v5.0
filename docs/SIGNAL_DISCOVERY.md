# V5.5.1 Signal Discovery

## 目标

V5.5.1 将 V5.5 的“稳定性/增量信息”从纯函数层推进到真实历史数据运行器，针对每个基础信号建立可审计的 Signal Research Registry。

输入为 macau_history.json 的 point-in-time OOS folds；本版本不重新拟合预测模型，也不使用未来结果选择信号。

## 分析内容

- Hit@1 / Hit@3 / Hit@6
- multiclass LogLoss / Brier / ECE
- Information Gain：log(p(actual) / baseline)
- 30/60/120/180/300 期滚动命中率
- 相邻窗口衰减：多个窗口而非只比较最后两段
- 年度/period 稳定性
- OOS scalar score：每折取 p(actual)，用于信号相关性与冗余分析
- Pearson 相关与 redundancy groups
- Bootstrap 95% CI
- baseline p-value
- 描述性分类：STABLE_OOS / DECAYING / UNSTABLE / REDUNDANT / NO_EVIDENCE

## 运行

~~~bash
python v55_1_signal_discovery_run.py --input macau_history.json --output-dir reports/v5.5.1
~~~

输出：

- reports/v5.5.1/signal_registry.json
- reports/v5.5.1/signal_metrics.csv
- reports/v5.5.1/report.md

## 当前历史数据

仓库当前历史文件为 999 期，时间覆盖 2024-01-04 至 2026-09-24。V5.5.1 runner 会使用该文件，不进行网络刷新。

## 研究边界

这里的“alpha”只表示历史 OOS 的增量信息诊断，不等价于未来可预测性。特别是六合彩这类近随机过程，统计显著性、滚动稳定性和历史命中率都不能单独证明未来存在稳定优势。

下一阶段只有在真实运行结果显示存在可重复、低冗余且跨时期稳定的历史信息后，才进入 V5.6 Regime Detection / Adaptive Ensemble；否则应优先做数据质量、基线和多重检验审计，而不是继续堆叠模型。
