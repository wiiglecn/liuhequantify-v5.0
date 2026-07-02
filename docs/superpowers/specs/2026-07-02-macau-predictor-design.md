# 澳门六合彩开奖记录分析与预测软件 设计规格

- 日期: 2026-07-02
- 语言: Python 3
- 新建独立程序, 不改动现有 `lottery_analyzer.py`

## 1. 背景与目标

依据 `https://macaujc.com`(澳门六合彩) 历史开奖记录, 采用统计模型 + 大模型反推理, 给出下一期开奖的三组预测, 评估各组命中率, 并罗列反推理出的候选数学模型。

### 1.1 数据源(已验证可用)

macaujc.com 是 Vue SPA, 数据由 JSON API 提供:

- 最新一期: `https://macaumarksix.com/api/macaujc2.com` → JSON 数组(单元素)
- 按年历史: `https://history.macaumarksix.com/history/macaujc2/y/{year}` → `{result, code, data:[...]}`

记录字段:
- `expect`: 期号, 如 `2026182`
- `openTime`: 开奖时间, 如 `2026-07-01 21:32:32`
- `openCode`: 开奖号码, 如 `15,23,09,45,24,39,41` — 6 个平码 + 末位为特码, 范围 1–49
- `wave`: 波色, 逗号分隔, 如 `blue,red,blue,red,red,green,blue`
- `zodiac`: 生肖, 逗号分隔, 如 `龍,猴,狗,狗,羊,龍,虎`

开奖频率: 每日一期, 约 365 期/年。样本取近 3 年(2024–2026), 约 1000+ 期。

## 2. 决策记录

| 决策点 | 选择 |
|--------|------|
| 大模型接入 | OpenAI 兼容 API(DeepSeek 等), 配置走 config.ini; Key 缺失时优雅降级 |
| 预测目标 | 完整 7 个号码(6 平码 + 1 特码), 范围 1–49 |
| 科学诚实度 | 加入随机性免责声明; 命中率基于真实历史回测而非空口评估 |

## 3. 架构(4 层, 独立可测)

```
数据层(data_fetcher) → 统计分析层(analysis) → 大模型反推理层(llm_reasoner)
                                          → 预测与回测层(predictor) → CLI 报告
```

主入口 `macau_predictor.py` 串联各层。

### 3.1 文件结构

```
liuhequantify/
├── macau_predictor.py   # 主入口, CLI 报告
├── data_fetcher.py      # 数据抓取与解析 + 本地缓存
├── analysis.py          # 统计分析 / 模型反推
├── llm_reasoner.py      # 大模型反推理
├── predictor.py         # 三组预测 + 回测
├── config.ini           # LLM 配置(base_url / api_key / model)
└── macau_history.json   # 数据缓存
```

## 4. 各模块设计

### 4.1 data_fetcher.py — 数据层

接口:
- `fetch_latest() -> list[dict]`: 抓取最新一期
- `fetch_year(year: int) -> list[dict]`: 抓取某年历史
- `load_history(years: list[int], refresh=False) -> list[Record]`: 加载多期, 合并去重(按 expect), 按期号排序; 本地缓存到 `macau_history.json`; `refresh=True` 强制重新抓取
- `Record` 数据类: `expect, open_time, regular(6 个 int), special(int), waves, zodiacs`

实现要点:
- SSL 跳过证书验证(与现有脚本一致)
- UA 伪装, Referer 设为 macaujc.com
- 解析 `openCode` 拆为 regular(前6) + special(末1), int 化
- 合并多年数据后按 expect 升序排序
- 去重: 同 expect 保留后抓到的(更新)

### 4.2 analysis.py — 统计分析层(模型反推)

对历史记录跑以下统计检验, 每个模型输出"拟合度评分" 0–1, 作为反推理候选清单:

| 模型 | 方法 | 含义 |
|------|------|------|
| 均匀分布(χ²) | 卡方检验 49 个号码频率 | 检验是否纯随机 |
| 频率/冷热号 | 各号频次、遗漏期数 | 热号/冷号 |
| 马尔可夫一阶 | 特码状态转移矩阵 | 状态依赖性 |
| 马尔可夫二阶 | 特码二阶转移 | 更强依赖 |
| 自相关/周期 | lag=1..7 自相关 + FFT | 是否周期性 |
| 和值正态拟合 | 6 码和值, 拟合正态 | 和值分布 |
| 奇偶/大小/区间 | 各维度分布 | 结构特征 |
| 尾数分布 | 个位分布 | 尾数规律 |
| 生肖/波色分布 | 频率 | 生肖波色偏好 |

接口:
- `analyze(records) -> AnalysisReport`, 内含各模型摘要 + 评分
- `summarize_for_llm(report) -> str`: 压缩为大模型 prompt 用的文本摘要

### 4.3 llm_reasoner.py — 大模型反推理层

接口:
- `load_config() -> dict | None`: 读 config.ini; Key/base_url 缺失返回 None
- `reason(analysis_text, recent_draws) -> LLMResult`: 调用 OpenAI 兼容接口
  - prompt 让模型: (1) 反推理最可能的数学模型并罗列; (2) 给下一期预测推理与理由
- `LLMResult`: `inferred_models(list[str]), predicted_set(list[int]), reasoning(str)`

降级: 无配置或调用失败 → 返回 None, 主流程跳过 LLM 分支, 仅用统计层。

实现: 用 `urllib`(不引入额外依赖), POST `{base_url}/chat/completions`, 传 model/messages。出错打 warning, 不中断。

### 4.4 predictor.py — 预测与回测层

三组预测(每组 = 6 平码 + 1 特码):

| 组 | 策略 |
|----|------|
| A | 频率/冷热加权: 热号权重 + 生肖约束 + 和值约束抽样 |
| B | 马尔可夫转移: 基于特码一阶转移矩阵 + 和值正态抽样 |
| C | 大模型集成: LLM 预测 + 统计融合; 无 LLM 时退化为遗漏值回归策略 |

抽样约束(各组通用, 保证合理性):
- 6 个平码互不相同, 范围 1–49
- 特码可与平码重复(六合彩规则允许), 但实际常独立; 默认特码从全候选中按策略抽
- 和值落在历史和值均值 ± 1 标准差区间内(重试抽样)

命中率评估(关键):
- `backtest(records, group_func, n=30)`: 用最近 n 期做留一回测。对每一期 t, 仅用 t 之前的数据生成预测, 对比 t 真实开奖:
  - 特码命中(预测特码 == 真实特码)
  - 平码命中数(|预测平码 ∩ 真实平码|)
- 汇总各组平均特码命中率、平均平码命中数
- 按回测命中率排序, 报告"哪组命中率更高"

接口:
- `predict_all(records, analysis, llm_result) -> list[PredictionGroup]`
- `PredictionGroup`: `name, regular(6), special, strategy_desc, backtest_special_hit_rate, backtest_regular_avg_hits`

### 4.5 macau_predictor.py — 主入口 / CLI

流程:
1. 加载历史(`--refresh` 可强制刷新)
2. 统计分析
3. (可选)大模型反推理
4. 三组预测 + 回测
5. 输出报告

报告内容:
- 数据概览(期数、时间范围、最近一期)
- 反推理出的数学模型清单(含拟合度评分)
- 三组预测号码(标注特码)+ 各组回测命中率 + 推荐组
- 科学免责声明

CLI: `python -X utf8 macau_predictor.py [--refresh] [--years 2024,2025,2026] [--backtest 30]`

免责声明文本(输出末尾):
> 科学提示: 彩票开奖本质为独立随机事件, 任何模型的真实命中率理论上均接近随机概率(特码命中概率约 1/49≈2%, 单个平码命中约 6/49)。本软件的命中率评估基于历史回测, 不代表未来表现, 结果仅供数学/统计研究与方法验证, 不构成任何投注建议。

## 5. 错误处理

- 网络抓取失败: 重试 2 次, 仍失败则用本地缓存; 缓存也没有则报错退出
- JSON 解析失败: 记录原始响应, 跳过该条
- LLM 调用失败/无配置: 降级, 仅统计层
- 回测样本不足: 警告并减少 n

## 6. 测试

- data_fetcher: mock 响应测解析、去重、排序
- analysis: 用已知序列测卡方/转移矩阵正确性
- predictor: 测三组抽样约束(不重复、和值范围)、回测逻辑正确性
- llm_reasoner: 无配置时返回 None; mock API 测响应解析

## 7. 依赖

仅标准库(urllib, ssl, json, math, collections, configparser, statistics, re)。无第三方依赖。
