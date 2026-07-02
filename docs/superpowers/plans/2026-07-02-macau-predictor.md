# 澳门六合彩开奖记录分析与预测软件 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从 macaujc.com 抓取澳门六合彩历史开奖记录，用统计模型 + 大模型反推理，给出下一期三组预测并回测评估命中率。

**Architecture:** 4 层独立模块 — 数据层(data_fetcher) → 统计分析层(analysis) → 大模型反推理层(llm_reasoner) → 预测回测层(predictor)，由 macau_predictor.py 串联。仅用 Python 标准库。

**Tech Stack:** Python 3 标准库 (urllib, ssl, json, math, collections, configparser, statistics, re), OpenAI 兼容 HTTP API, 内置 unittest 测试框架。

**Spec:** `docs/superpowers/specs/2026-07-02-macau-predictor-design.md`

---

## 文件结构

| 文件 | 职责 |
|------|------|
| `data_fetcher.py` | 抓取/解析 macaujc API，标准化 Record，本地缓存 |
| `analysis.py` | 统计检验与模型反推，输出 AnalysisReport |
| `llm_reasoner.py` | 读 config.ini，调 OpenAI 兼容接口反推理 |
| `predictor.py` | 三组预测 + 留一回测 |
| `macau_predictor.py` | 主入口，CLI 报告 |
| `config.ini` | LLM 配置模板 |
| `tests/test_data_fetcher.py` | 数据层测试 |
| `tests/test_analysis.py` | 分析层测试 |
| `tests/test_predictor.py` | 预测回测测试 |
| `tests/test_llm_reasoner.py` | LLM 层测试 |
| `macau_history.json` | 数据缓存（运行时生成） |

约定：所有源文件顶部含 Windows UTF-8 stdout 重配（与现有 lottery_analyzer.py 一致）。

---

## Task 1: 项目骨架与测试目录

**Files:**
- Create: `tests/__init__.py` (空文件)
- Create: `tests/test_smoke.py`

- [ ] **Step 1: 创建测试目录与占位测试**

`tests/__init__.py` 为空文件。

`tests/test_smoke.py`:
```python
import unittest

class TestSmoke(unittest.TestCase):
    def test_python_runs(self):
        self.assertEqual(1 + 1, 2)

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试验证环境**

Run: `python -X utf8 -m unittest tests.test_smoke -v`
Expected: PASS (test_python_runs)

- [ ] **Step 3: 提交**

```bash
git add tests/
git commit -m "chore: 添加测试目录骨架"
```

---

## Task 2: data_fetcher — Record 与解析

**Files:**
- Create: `data_fetcher.py`
- Test: `tests/test_data_fetcher.py`

- [ ] **Step 1: 写失败测试 — 解析单条 API 记录为 Record**

`tests/test_data_fetcher.py`:
```python
import unittest
from data_fetcher import parse_record, Record

class TestParseRecord(unittest.TestCase):
    def test_parse_standard_record(self):
        raw = {
            "expect": "2026182",
            "openTime": "2026-07-01 21:32:32",
            "openCode": "15,23,09,45,24,39,41",
            "wave": "blue,red,blue,red,red,green,blue",
            "zodiac": "龍,猴,狗,狗,羊,龍,虎",
        }
        rec = parse_record(raw)
        self.assertIsInstance(rec, Record)
        self.assertEqual(rec.expect, "2026182")
        self.assertEqual(rec.open_time, "2026-07-01 21:32:32")
        self.assertEqual(rec.regular, [15, 23, 9, 45, 24, 39])
        self.assertEqual(rec.special, 41)
        self.assertEqual(len(rec.waves), 7)
        self.assertEqual(len(rec.zodiacs), 7)

    def test_parse_pads_short_opencode(self):
        raw = {"expect": "1", "openTime": "t", "openCode": "5,10",
               "wave": "blue,red", "zodiac": "鼠,牛"}
        rec = parse_record(raw)
        self.assertEqual(rec.regular, [5, 10])
        self.assertEqual(rec.special, 10)  # 不足7个时，末位即特码

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试验证失败**

Run: `python -X utf8 -m unittest tests.test_data_fetcher -v`
Expected: FAIL (ImportError: No module named 'data_fetcher')

- [ ] **Step 3: 实现 data_fetcher.py 的 Record 与 parse_record**

`data_fetcher.py`:
```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""澳门六合彩数据抓取与解析层"""
import os
import sys
import json
import ssl
import urllib.request
from dataclasses import dataclass, field

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        os.system("chcp 65001 >nul 2>&1")

LATEST_URL = "https://macaumarksix.com/api/macaujc2.com"
HISTORY_URL = "https://history.macaumarksix.com/history/macaujc2/y/{year}"
CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "macau_history.json")


@dataclass
class Record:
    expect: str
    open_time: str
    regular: list  # 6 个平码 int
    special: int   # 特码
    waves: list = field(default_factory=list)
    zodiacs: list = field(default_factory=list)


def parse_record(raw: dict) -> Record:
    """将 API 原始 dict 解析为 Record。openCode 前6位为平码，末位为特码。"""
    codes = [int(x) for x in str(raw.get("openCode", "")).split(",") if x.strip()]
    waves = [w for w in str(raw.get("wave", "")).split(",") if w.strip()]
    zodiacs = [z for z in str(raw.get("zodiac", "")).split(",") if z.strip()]
    if not codes:
        codes = [0]
    if len(codes) >= 7:
        regular, special = codes[:6], codes[6]
    else:
        regular = codes[:-1] if len(codes) > 1 else codes
        special = codes[-1]
    return Record(
        expect=str(raw.get("expect", "")),
        open_time=str(raw.get("openTime", "")),
        regular=regular,
        special=special,
        waves=waves,
        zodiacs=zodiacs,
    )
```

- [ ] **Step 4: 运行测试验证通过**

Run: `python -X utf8 -m unittest tests.test_data_fetcher -v`
Expected: PASS (2 tests)

- [ ] **Step 5: 提交**

```bash
git add data_fetcher.py tests/test_data_fetcher.py
git commit -m "feat: data_fetcher Record 解析"
```

---

## Task 3: data_fetcher — 抓取与缓存

**Files:**
- Modify: `data_fetcher.py`
- Test: `tests/test_data_fetcher.py` (追加)

- [ ] **Step 1: 写失败测试 — 合并去重排序、缓存读写**

追加到 `tests/test_data_fetcher.py`（在 import 区加 `from data_fetcher import merge_records, save_cache, load_cache, Record`，并在文件内新增类）:
```python
class TestMergeAndCache(unittest.TestCase):
    def test_merge_dedup_sort(self):
        a = [Record("2026182", "t1", [1,2,3,4,5,6], 7),
             Record("2026180", "t0", [1,2,3,4,5,6], 8)]
        b = [Record("2026181", "t1b", [2,2,3,4,5,6], 9),
             Record("2026182", "t1new", [3,3,3,4,5,6], 10)]  # 重复, 后者覆盖
        merged = merge_records([a, b])
        self.assertEqual([r.expect for r in merged], ["2026180", "2026181", "2026182"])
        self.assertEqual(merged[-1].regular, [3,3,3,4,5,6])  # b 覆盖 a

    def test_cache_roundtrip(self, tmp_path=None):
        import tempfile, os
        d = tempfile.mkdtemp()
        path = os.path.join(d, "c.json")
        recs = [Record("1", "t", [1,2,3,4,5,6], 7, ["red"], ["鼠"])]
        save_cache(recs, path)
        loaded = load_cache(path)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].expect, "1")
        self.assertEqual(loaded[0].regular, [1,2,3,4,5,6])

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试验证失败**

Run: `python -X utf8 -m unittest tests.test_data_fetcher.TestMergeAndCache -v`
Expected: FAIL (ImportError: cannot import name 'merge_records')

- [ ] **Step 3: 实现抓取/合并/缓存函数**

追加到 `data_fetcher.py`:
```python
def _ssl_context():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _fetch_json(url: str, retries: int = 2):
    last_err = None
    for _ in range(retries + 1):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0", "Referer": "https://macaujc.com/"}
            )
            with urllib.request.urlopen(req, context=_ssl_context(), timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8", "ignore"))
        except Exception as e:
            last_err = e
    raise last_err


def fetch_latest() -> list:
    """抓取最新一期, 返回原始 dict 列表。"""
    data = _fetch_json(LATEST_URL)
    return data if isinstance(data, list) else []


def fetch_year(year: int) -> list:
    """抓取某年历史, 返回原始 dict 列表(已从 {result,data} 解包)。"""
    data = _fetch_json(HISTORY_URL.format(year=year))
    if isinstance(data, dict) and "data" in data:
        return data["data"] or []
    if isinstance(data, list):
        return data
    return []


def merge_records(record_lists: list) -> list:
    """合并多份 Record 列表, 按 expect 去重(后者覆盖)并升序排序。"""
    by_expect = {}
    for lst in record_lists:
        for r in lst:
            by_expect[r.expect] = r
    return [by_expect[k] for k in sorted(by_expect.keys())]


def save_cache(records: list, path: str = CACHE_FILE):
    with open(path, "w", encoding="utf-8") as f:
        json.dump([{"expect": r.expect, "openTime": r.open_time,
                    "regular": r.regular, "special": r.special,
                    "waves": r.waves, "zodiacs": r.zodiacs} for r in records],
                   f, ensure_ascii=False, indent=2)


def load_cache(path: str = CACHE_FILE) -> list:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)
    return [Record(expect=it["expect"], open_time=it["openTime"],
                   regular=it["regular"], special=it["special"],
                   waves=it.get("waves", []), zodiacs=it.get("zodiacs", []))
            for it in items]


def load_history(years: list, refresh: bool = False) -> list:
    """加载多年记录, 优先用缓存, refresh=True 强制重新抓取。"""
    cached = load_cache() if not refresh else []
    if cached and not refresh:
        # 校验缓存是否覆盖所需年份
        return cached
    lists = []
    for y in years:
        try:
            lists.append([parse_record(r) for r in fetch_year(y)])
        except Exception as e:
            print(f"  ! 抓取 {y} 年失败: {e}")
    try:
        lists.append([parse_record(r) for r in fetch_latest()])
    except Exception as e:
        print(f"  ! 抓取最新期失败: {e}")
    merged = merge_records(lists)
    if merged:
        save_cache(merged)
    return merged
```

- [ ] **Step 4: 运行测试验证通过**

Run: `python -X utf8 -m unittest tests.test_data_fetcher -v`
Expected: PASS (4 tests)

- [ ] **Step 5: 提交**

```bash
git add data_fetcher.py tests/test_data_fetcher.py
git commit -m "feat: data_fetcher 抓取/合并/缓存"
```

---

## Task 4: analysis — 频率与卡方均匀性检验

**Files:**
- Create: `analysis.py`
- Test: `tests/test_analysis.py`

- [ ] **Step 1: 写失败测试 — 频率统计与卡方**

`tests/test_analysis.py`:
```python
import unittest
from data_fetcher import Record
from analysis import number_frequency, chi_square_uniform

def mk(expect, special, regular):
    return Record(expect, "t", regular, special)

class TestFrequency(unittest.TestCase):
    def test_number_frequency_counts_all_seven(self):
        recs = [mk("1", 7, [1,2,3,4,5,6]), mk("2", 7, [1,2,3,4,5,6])]
        freq = number_frequency(recs)
        # 每期 7 个号, 2 期 = 14 次出现; 1-6 各 2 次, 7 出现 2 次
        self.assertEqual(freq[1], 2)
        self.assertEqual(freq[7], 2)
        self.assertEqual(sum(freq.values()), 14)

    def test_chi_square_uniform_uniform_input(self):
        # 完全均匀: 49 个号各出现 10 次
        recs = []
        for i in range(10):
            recs.append(mk(str(i), 49, list(range(1, 7))))
            recs.append(mk(str(i) + "b", 49, list(range(7, 13)) * 1))
        # 简化: 直接用频率测卡方接口
        freq = {n: 10 for n in range(1, 50)}
        chi2, p = chi_square_uniform(freq)
        self.assertAlmostEqual(chi2, 0.0, places=6)

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试验证失败**

Run: `python -X utf8 -m unittest tests.test_analysis -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: 实现 analysis.py 频率与卡方**

`analysis.py`:
```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统计分析层 — 模型反推"""
import os
import sys
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        os.system("chcp 65001 >nul 2>&1")


def _all_numbers(rec):
    """一期全部 7 个号码(平码+特码)。"""
    return list(rec.regular) + [rec.special]


def number_frequency(records) -> dict:
    """统计 1-49 各号码出现次数(含特码)。"""
    c = Counter()
    for r in records:
        c.update(_all_numbers(r))
    return {n: c.get(n, 0) for n in range(1, 50)}


def chi_square_uniform(freq: dict):
    """卡方均匀性检验。返回 (chi2, p_value)。自由度=48。"""
    observed = [freq.get(n, 0) for n in range(1, 50)]
    total = sum(observed)
    if total == 0:
        return 0.0, 1.0
    expected = total / 49.0
    chi2 = sum((o - expected) ** 2 / expected for o in observed)
    # 用不完全伽马函数近似 p 值 (自由度 df=48)
    df = 48
    try:
        p = 1.0 - _gammainc(df / 2.0, chi2 / 2.0)
    except Exception:
        p = float("nan")
    return chi2, p


def _gammainc_lower(s, x):
    """下不完全伽马函数 (级数展开, x 适中时稳定)。"""
    if x < 0:
        return 0.0
    if x == 0:
        return 0.0
    # 级数
    term = 1.0 / s
    total = term
    for n in range(1, 1000):
        term *= x / (s + n)
        total += term
        if abs(term) < 1e-12:
            break
    return total * (x ** s) * math.exp(-x) / math.gamma(s)


def _gammainc(s, x):
    """正则化上不完全伽马 P(s,x) = lower/Γ(s)。"""
    if x < 0:
        return 0.0
    if x == 0:
        return 0.0
    return _gammainc_lower(s, x) / math.gamma(s)
```

> 注意: `chi_square_uniform` 返回的 p 用上不完全伽马定义; 上面 `_gammainc` 返回的是 lower/Γ(s) = P(s,x) = CDF。卡方 p-value = 1 - CDF = 1 - P(df/2, chi2/2)。已修正: `p = 1.0 - _gammainc(...)`。

- [ ] **Step 4: 运行测试验证通过**

Run: `python -X utf8 -m unittest tests.test_analysis -v`
Expected: PASS (2 tests)

- [ ] **Step 5: 提交**

```bash
git add analysis.py tests/test_analysis.py
git commit -m "feat: analysis 频率与卡方均匀性检验"
```

---

## Task 5: analysis — 冷热号、遗漏值、和值正态

**Files:**
- Modify: `analysis.py`
- Test: `tests/test_analysis.py` (追加)

- [ ] **Step 1: 写失败测试 — 遗漏值与和值统计**

追加到 `tests/test_analysis.py`（import 加 `from analysis import gap_stats, sum_stats, hot_cold`）:
```python
class TestGapAndSum(unittest.TestCase):
    def test_gap_stats(self):
        recs = [mk("1", 7, [1,2,3,4,5,6]),
                mk("2", 8, [1,2,3,4,5,6]),
                mk("3", 7, [1,2,3,4,5,6])]
        gap = gap_stats(recs)
        # 号码 7 出现在第1、3期, 当前遗漏(自最后一次出现)=0, 最大遗漏=1
        self.assertEqual(gap[7]["current"], 0)
        self.assertEqual(gap[7]["max"], 1)
        # 号码 8 只在第2期, 当前遗漏=1
        self.assertEqual(gap[8]["current"], 1)

    def test_sum_stats(self):
        recs = [mk("1", 7, [1,2,3,4,5,6]), mk("2", 49, [10,20,30,40,44,48])]
        s = sum_stats(recs)
        # 平码和值: 21 与 192
        self.assertAlmostEqual(s["mean"], (21 + 192) / 2)
        self.assertIn("std", s)

    def test_hot_cold(self):
        recs = [mk("1", 7, [1,1,1,1,1,1])]  # 1 出现 6 次, 7 出现 1 次
        hot, cold = hot_cold(number_frequency(recs), topn=3)
        self.assertIn(1, hot)
        self.assertIn(49, cold)  # 未出现的号是冷号
```

- [ ] **Step 2: 运行测试验证失败**

Run: `python -X utf8 -m unittest tests.test_analysis.TestGapAndSum -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: 实现遗漏/和值/冷热号**

追加到 `analysis.py`:
```python
def gap_stats(records) -> dict:
    """每个号码的遗漏统计: current(当前遗漏期数), max(历史最大遗漏)。"""
    last_seen = {}
    max_gap = {n: 0 for n in range(1, 50)}
    current_gap = {n: 0 for n in range(1, 50)}
    for i, r in enumerate(records):
        nums = set(_all_numbers(r))
        for n in range(1, 50):
            if n in nums:
                gap = i - last_seen.get(n, i)
                if gap > max_gap[n]:
                    max_gap[n] = gap
                last_seen[n] = i
                current_gap[n] = 0
            else:
                current_gap[n] = i - last_seen.get(n, -1) if n in last_seen else i + 1
    # 最终 current: 最后一期到现在的遗漏
    total = len(records)
    for n in range(1, 50):
        if n in last_seen:
            current_gap[n] = total - 1 - last_seen[n]
        else:
            current_gap[n] = total
    return {n: {"current": current_gap[n], "max": max_gap[n]} for n in range(1, 50)}


def sum_stats(records) -> dict:
    """6 平码和值的均值/标准差/最小/最大。"""
    sums = [sum(r.regular) for r in records if len(r.regular) == 6]
    if not sums:
        return {"mean": 0, "std": 0, "min": 0, "max": 0}
    mean = sum(sums) / len(sums)
    var = sum((s - mean) ** 2 for s in sums) / len(sums)
    return {"mean": mean, "std": math.sqrt(var), "min": min(sums), "max": max(sums)}


def hot_cold(freq: dict, topn: int = 10):
    """返回热号(频次最高)与冷号(频次最低)各 topn 个。"""
    items = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)
    hot = [n for n, _ in items[:topn]]
    cold = [n for n, _ in items[-topn:]]
    return hot, cold
```

- [ ] **Step 4: 运行测试验证通过**

Run: `python -X utf8 -m unittest tests.test_analysis -v`
Expected: PASS (5 tests)

- [ ] **Step 5: 提交**

```bash
git add analysis.py tests/test_analysis.py
git commit -m "feat: analysis 遗漏/和值/冷热号"
```

---

## Task 6: analysis — 马尔可夫转移与自相关

**Files:**
- Modify: `analysis.py`
- Test: `tests/test_analysis.py` (追加)

- [ ] **Step 1: 写失败测试 — 马尔可夫转移与自相关**

追加到 `tests/test_analysis.py`（import 加 `from analysis import markov_transition, autocorrelation, build_report`）:
```python
class TestMarkovAndAc(unittest.TestCase):
    def test_markov_transition(self):
        recs = [mk("1", 1, [0]*6), mk("2", 2, [0]*6), mk("3", 1, [0]*6)]
        trans = markov_transition(recs)
        # 特码序列 1->2->1, 从 1 转到 2 出现 1 次, 从 2 转到 1 出现 1 次
        self.assertAlmostEqual(trans[1][2], 0.5)   # 1 后出现 2 一次, 出现 1(末位不计转移) — 1->2 与 1->1
        self.assertTrue(trans[1][2] > 0)

    def test_autocorrelation_constant_series(self):
        # 常数序列自相关应为 0(去均值后分母为0, 返回 0)
        ac = autocorrelation([5,5,5,5,5], lag=1)
        self.assertEqual(ac, 0.0)

    def test_build_report_returns_dict(self):
        recs = [mk(str(i), (i % 49) + 1, [1,2,3,4,5,6]) for i in range(20)]
        rep = build_report(recs)
        self.assertIn("freq", rep)
        self.assertIn("chi2", rep)
        self.assertIn("sum", rep)
        self.assertIn("markov", rep)
        self.assertIn("models", rep)
```

- [ ] **Step 2: 运行测试验证失败**

Run: `python -X utf8 -m unittest tests.test_analysis.TestMarkovAndAc -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: 实现马尔可夫与自相关 + build_report**

追加到 `analysis.py`:
```python
def markov_transition(records):
    """特码一阶转移矩阵(归一化行概率)。返回 {from: {to: prob}}。"""
    specials = [r.special for r in records]
    trans = defaultdict(lambda: defaultdict(int))
    for a, b in zip(specials[:-1], specials[1:]):
        trans[a][b] += 1
    result = {}
    for a, nxt in trans.items():
        total = sum(nxt.values())
        result[a] = {b: c / total for b, c in nxt.items()}
    return result


def markov_transition_2(records):
    """特码二阶转移。返回 {(a,b): {to: prob}}。"""
    specials = [r.special for r in records]
    trans = defaultdict(lambda: defaultdict(int))
    for i in range(len(specials) - 2):
        key = (specials[i], specials[i + 1])
        trans[key][specials[i + 2]] += 1
    result = {}
    for key, nxt in trans.items():
        total = sum(nxt.values())
        result[key] = {b: c / total for b, c in nxt.items()}
    return result


def autocorrelation(series: list, lag: int = 1) -> float:
    """序列滞后 lag 的自相关系数。常数序列返回 0。"""
    n = len(series)
    if n <= lag:
        return 0.0
    mean = sum(series) / n
    num = sum((series[i] - mean) * (series[i + lag] - mean) for i in range(n - lag))
    den = sum((x - mean) ** 2 for x in series)
    if den == 0:
        return 0.0
    return num / den


def _zodiac_wave_freq(records):
    zc = Counter()
    wc = Counter()
    for r in records:
        zc.update(r.zodiacs)
        wc.update(r.waves)
    return dict(zc), dict(wc)


def _odd_even_big_small(records):
    odd = even = big = small = 0
    for r in records:
        for n in _all_numbers(r):
            if n % 2 == 0:
                even += 1
            else:
                odd += 1
            if n >= 25:
                big += 1
            else:
                small += 1
    return {"odd": odd, "even": even, "big": big, "small": small}


def build_report(records) -> dict:
    """聚合所有统计, 并给出反推理候选模型清单。"""
    freq = number_frequency(records)
    chi2, p = chi_square_uniform(freq)
    gap = gap_stats(records)
    s = sum_stats(records)
    mk1 = markov_transition(records)
    mk2 = markov_transition_2(records)
    specials = [r.special for r in records]
    ac1 = autocorrelation(specials, 1)
    ac7 = autocorrelation(specials, 7)
    zodiac, wave = _zodiac_wave_freq(records)
    oebs = _odd_even_big_small(records)

    # 模型反推评分(0-1, 越高越可能"解释"该序列)
    models = []
    # 1. 均匀随机: p 值越大越像均匀随机
    models.append(("均匀分布(纯随机)", min(1.0, max(0.0, p)), "卡方 p=%.4f" % p))
    # 2. 马尔可夫: 转移矩阵熵越低越有依赖
    import math as _m
    mk_entropy = 0.0
    for a, nxt in mk1.items():
        for prob in nxt.values():
            if prob > 0:
                mk_entropy -= prob * _m.log2(prob)
    max_entropy = _m.log2(49)
    mk_score = 1.0 - (mk_entropy / max_entropy) if max_entropy else 0
    models.append(("马尔可夫一阶转移", max(0.0, min(1.0, mk_score)), "转移熵=%.3f" % mk_entropy))
    # 3. 自相关周期: |ac| 越大越有周期性
    models.append(("自相关/周期性", min(1.0, abs(ac1) + abs(ac7) / 2), "ac1=%.3f ac7=%.3f" % (ac1, ac7)))
    # 4. 和值正态: 样本量足够时假设近似正态
    cv = (s["std"] / s["mean"]) if s["mean"] else 0
    models.append(("和值正态分布", max(0.0, 1.0 - cv), "均值=%.1f std=%.1f" % (s["mean"], s["std"])))
    models.append(("频率/冷热号偏置", min(1.0, chi2 / 200.0), "chi2=%.1f" % chi2))

    return {
        "freq": freq, "chi2": (chi2, p), "gap": gap, "sum": s,
        "markov": mk1, "markov2": mk2, "ac": (ac1, ac7),
        "zodiac": zodiac, "wave": wave, "oebs": oebs,
        "models": models,
    }
```

- [ ] **Step 4: 运行测试验证通过**

Run: `python -X utf8 -m unittest tests.test_analysis -v`
Expected: PASS (8 tests)

- [ ] **Step 5: 提交**

```bash
git add analysis.py tests/test_analysis.py
git commit -m "feat: analysis 马尔可夫/自相关/报告聚合"
```

---

## Task 7: analysis — LLM 摘要生成

**Files:**
- Modify: `analysis.py`
- Test: `tests/test_analysis.py` (追加)

- [ ] **Step 1: 写失败测试 — summarize_for_llm 含关键信息**

追加到 `tests/test_analysis.py`（import 加 `from analysis import summarize_for_llm`）:
```python
class TestSummarize(unittest.TestCase):
    def test_summary_contains_models_and_stats(self):
        recs = [mk(str(i), (i % 49) + 1, [1,2,3,4,5,6]) for i in range(20)]
        rep = build_report(recs)
        text = summarize_for_llm(rep, recent_n=5)
        self.assertIn("数学模型", text)
        self.assertIn("卡方", text)
        self.assertIn("特码序列", text)

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试验证失败**

Run: `python -X utf8 -m unittest tests.test_analysis.TestSummarize -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: 实现 summarize_for_llm**

追加到 `analysis.py`:
```python
def summarize_for_llm(report: dict, recent_n: int = 10) -> str:
    """将 AnalysisReport 压缩为大模型 prompt 摘要文本。"""
    freq = report["freq"]
    chi2, p = report["chi2"]
    s = report["sum"]
    ac1, ac7 = report["ac"]
    lines = []
    lines.append("== 澳门六合彩历史开奖统计分析摘要 ==")
    lines.append("【反推理候选数学模型】")
    for name, score, detail in report["models"]:
        lines.append(f"  - {name}: 拟合度={score:.3f} ({detail})")
    lines.append(f"【均匀性】卡方={chi2:.2f}, p={p:.4f} (p>0.05 表示不拒绝均匀分布)")
    lines.append(f"【和值】均值={s['mean']:.1f}, std={s['std']:.1f}, 区间=[{s['min']},{s['max']}]")
    lines.append(f"【自相关】lag1={ac1:.3f}, lag7={ac7:.3f}")
    # 冷热号 top10
    items = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)
    hot = [str(n) for n, _ in items[:10]]
    cold = [str(n) for n, _ in items[-10:]]
    lines.append(f"【热号Top10】{','.join(hot)}")
    lines.append(f"【冷号Top10】{','.join(cold)}")
    oebs = report["oebs"]
    lines.append(f"【奇偶/大小】奇={oebs['odd']} 偶={oebs['even']} 大(>=25)={oebs['big']} 小={oebs['small']}")
    lines.append(f"【生肖分布】{report['zodiac']}")
    lines.append(f"【波色分布】{report['wave']}")
    return "\n".join(lines)
```

- [ ] **Step 4: 运行测试验证通过**

Run: `python -X utf8 -m unittest tests.test_analysis -v`
Expected: PASS (9 tests)

- [ ] **Step 5: 提交**

```bash
git add analysis.py tests/test_analysis.py
git commit -m "feat: analysis LLM 摘要生成"
```

---

## Task 8: llm_reasoner — 配置加载与降级

**Files:**
- Create: `config.ini`
- Create: `llm_reasoner.py`
- Test: `tests/test_llm_reasoner.py`

- [ ] **Step 1: 写失败测试 — 配置缺失返回 None**

`tests/test_llm_reasoner.py`:
```python
import unittest, os, tempfile
from llm_reasoner import load_config

class TestConfig(unittest.TestCase):
    def test_missing_key_returns_none(self):
        d = tempfile.mkdtemp()
        path = os.path.join(d, "config.ini")
        with open(path, "w", encoding="utf-8") as f:
            f.write("[llm]\nbase_url=https://api.deepseek.com\nmodel=deepseek-chat\napi_key=\n")
        cfg = load_config(path)
        self.assertIsNone(cfg)  # api_key 为空 -> None

    def test_valid_config(self):
        d = tempfile.mkdtemp()
        path = os.path.join(d, "config.ini")
        with open(path, "w", encoding="utf-8") as f:
            f.write("[llm]\nbase_url=https://api.deepseek.com\nmodel=deepseek-chat\napi_key=sk-test\n")
        cfg = load_config(path)
        self.assertEqual(cfg["api_key"], "sk-test")
        self.assertEqual(cfg["model"], "deepseek-chat")

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试验证失败**

Run: `python -X utf8 -m unittest tests.test_llm_reasoner -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: 实现 config.ini 与 llm_reasoner 配置加载**

`config.ini`:
```ini
[llm]
# OpenAI 兼容接口配置(DeepSeek / 通义 / Moonshot / OpenAI 等)
base_url = https://api.deepseek.com
api_key =
model = deepseek-chat
```

`llm_reasoner.py`:
```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""大模型反推理层 — OpenAI 兼容接口"""
import os
import sys
import json
import ssl
import urllib.request
import configparser
from dataclasses import dataclass

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        os.system("chcp 65001 >nul 2>&1")

DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.ini")


@dataclass
class LLMResult:
    inferred_models: list  # 反推理出的数学模型清单
    predicted_set: list    # LLM 预测的号码(可空)
    reasoning: str         # 推理过程文本


def load_config(path: str = DEFAULT_CONFIG):
    """读取 config.ini, 返回 {base_url, api_key, model}; 关键项缺失返回 None。"""
    if not os.path.exists(path):
        return None
    cp = configparser.ConfigParser()
    cp.read(path, encoding="utf-8")
    if "llm" not in cp:
        return None
    base_url = cp["llm"].get("base_url", "").strip()
    api_key = cp["llm"].get("api_key", "").strip()
    model = cp["llm"].get("model", "").strip()
    if not base_url or not api_key or not model:
        return None
    return {"base_url": base_url, "api_key": api_key, "model": model}
```

- [ ] **Step 4: 运行测试验证通过**

Run: `python -X utf8 -m unittest tests.test_llm_reasoner -v`
Expected: PASS (2 tests)

- [ ] **Step 5: 提交**

```bash
git add config.ini llm_reasoner.py tests/test_llm_reasoner.py
git commit -m "feat: llm_reasoner 配置加载与降级"
```

---

## Task 9: llm_reasoner — 调用与响应解析

**Files:**
- Modify: `llm_reasoner.py`
- Test: `tests/test_llm_reasoner.py` (追加)

- [ ] **Step 1: 写失败测试 — 解析 LLM 响应文本**

追加到 `tests/test_llm_reasoner.py`（import 加 `from llm_reasoner import parse_llm_response, build_prompt`）:
```python
class TestParseAndPrompt(unittest.TestCase):
    def test_parse_response_extracts_models_and_numbers(self):
        text = ("反推理数学模型:\n1. 马尔可夫链\n2. 均匀分布\n\n"
                "下期预测号码: 03,12,18,25,33,41 特码: 07\n推理: 基于转移矩阵")
        res = parse_llm_response(text)
        self.assertIn("马尔可夫链", res.inferred_models)
        self.assertIn("均匀分布", res.inferred_models)
        self.assertIn(7, res.predicted_set)
        self.assertIn(41, res.predicted_set)
        self.assertIn("转移矩阵", res.reasoning)

    def test_build_prompt_contains_summary(self):
        p = build_prompt("统计摘要内容XYZ", recent_specials=[7, 8, 9])
        self.assertIn("XYZ", p)
        self.assertIn("7", p)

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试验证失败**

Run: `python -X utf8 -m unittest tests.test_llm_reasoner.TestParseAndPrompt -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: 实现 build_prompt / parse_llm_response / reason**

追加到 `llm_reasoner.py`:
```python
import re

PROMPT_TEMPLATE = """你是博彩数学反推理分析师。下面是澳门六合彩历史开奖的统计分析摘要。
请完成两项任务:

任务一: 反推理出最可能产生该序列的数学模型(罗列 3-6 个), 每个简要说明依据。
任务二: 依据上述模型, 推理下一期最可能开出的 7 个号码(6平码+1特码, 范围1-49, 平码不重复)。

请严格按以下格式输出:
【反推理数学模型】
1. <模型名>: <依据>
2. <模型名>: <依据>
...
【下期预测】
平码: a,b,c,d,e,f
特码: g
【推理】
<简述推理过程>

{summary}

最近10期特码序列: {recent}
"""


def build_prompt(summary: str, recent_specials: list) -> str:
    recent = ",".join(str(x) for x in recent_specials[-10:])
    return PROMPT_TEMPLATE.format(summary=summary, recent=recent)


def parse_llm_response(text: str) -> LLMResult:
    """解析 LLM 文本响应为 LLMResult。容错: 缺失字段返回空。"""
    models = []
    m_section = re.search(r"【反推理数学模型】(.*?)(【下期预测】|【推理】|$)", text, re.S)
    if m_section:
        for line in m_section.group(1).splitlines():
            line = re.sub(r"^\s*\d+\.\s*", "", line.strip())
            if line and ("：" in line or ":" in line):
                name = re.split(r"[：:]", line, 1)[0].strip()
                if name:
                    models.append(name)
    predicted = []
    p_section = re.search(r"【下期预测】(.*?)(【推理】|$)", text, re.S)
    if p_section:
        nums = re.findall(r"\d{1,2}", p_section.group(1))
        predicted = [int(x) for x in nums if 1 <= int(x) <= 49]
    reasoning = ""
    r_section = re.search(r"【推理】(.*)", text, re.S)
    if r_section:
        reasoning = r_section.group(1).strip()
    if not reasoning:
        reasoning = text.strip()
    return LLMResult(inferred_models=models, predicted_set=predicted, reasoning=reasoning)


def _ssl_context():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def reason(summary: str, recent_specials: list, config: dict) -> LLMResult:
    """调用 OpenAI 兼容接口。失败抛异常, 由调用方降级。"""
    prompt = build_prompt(summary, recent_specials)
    url = config["base_url"].rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": config["model"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + config["api_key"]},
    )
    with urllib.request.urlopen(req, context=_ssl_context(), timeout=60) as resp:
        result = json.loads(resp.read().decode("utf-8", "ignore"))
    text = result["choices"][0]["message"]["content"]
    return parse_llm_response(text)
```

- [ ] **Step 4: 运行测试验证通过**

Run: `python -X utf8 -m unittest tests.test_llm_reasoner -v`
Expected: PASS (4 tests)

- [ ] **Step 5: 提交**

```bash
git add llm_reasoner.py tests/test_llm_reasoner.py
git commit -m "feat: llm_reasoner 调用与响应解析"
```

---

## Task 10: predictor — 采样约束工具

**Files:**
- Create: `predictor.py`
- Test: `tests/test_predictor.py`

- [ ] **Step 1: 写失败测试 — 加权采样与和值约束**

`tests/test_predictor.py`:
```python
import unittest, random
from predictor import weighted_sample, sample_with_sum_constraint

class TestSampling(unittest.TestCase):
    def test_weighted_sample_distinct_six(self):
        random.seed(42)
        weights = {n: 1.0 for n in range(1, 50)}
        weights[7] = 100.0  # 极高权重
        picked = weighted_sample(weights, k=6)
        self.assertEqual(len(picked), 6)
        self.assertEqual(len(set(picked)), 6)
        self.assertIn(7, picked)

    def test_sample_with_sum_constraint(self):
        random.seed(1)
        weights = {n: 1.0 for n in range(1, 50)}
        picked = sample_with_sum_constraint(weights, mean=150, std=30, k=6)
        self.assertEqual(len(set(picked)), 6)
        self.assertTrue(150 - 60 <= sum(picked) <= 150 + 60)

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试验证失败**

Run: `python -X utf8 -m unittest tests.test_predictor -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: 实现采样工具**

`predictor.py`:
```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""预测与回测层"""
import os
import sys
import random
from dataclasses import dataclass, field

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        os.system("chcp 65001 >nul 2>&1")


@dataclass
class PredictionGroup:
    name: str
    regular: list          # 6 个平码
    special: int           # 特码
    strategy: str          # 策略描述
    backtest_special_hit: float = 0.0   # 回测特码命中率
    backtest_regular_hits: float = 0.0  # 回测平均平码命中数


def weighted_sample(weights: dict, k: int = 6) -> list:
    """按权重无放回采样 k 个不同号码。"""
    pool = list(weights.keys())
    w = [weights[n] for n in pool]
    picked = []
    pool_copy = list(pool)
    w_copy = list(w)
    for _ in range(k):
        total = sum(w_copy)
        if total <= 0:
            choice = random.choice(pool_copy)
        else:
            r = random.random() * total
            cum = 0
            idx = 0
            for i, wi in enumerate(w_copy):
                cum += wi
                if r <= cum:
                    idx = i
                    break
            choice = pool_copy[idx]
        picked.append(choice)
        # 移除已选
        j = pool_copy.index(choice)
        pool_copy.pop(j)
        w_copy.pop(j)
    return picked


def sample_with_sum_constraint(weights: dict, mean: float, std: float,
                               k: int = 6, max_tries: int = 200) -> list:
    """采样 k 个, 和值需落在 [mean-2*std, mean+2*std]。"""
    lo, hi = mean - 2 * std, mean + 2 * std
    for _ in range(max_tries):
        picked = weighted_sample(weights, k=k)
        if lo <= sum(picked) <= hi:
            return picked
    return picked  # 实在不行返回最后一次
```

- [ ] **Step 4: 运行测试验证通过**

Run: `python -X utf8 -m unittest tests.test_predictor -v`
Expected: PASS (2 tests)

- [ ] **Step 5: 提交**

```bash
git add predictor.py tests/test_predictor.py
git commit -m "feat: predictor 采样约束工具"
```

---

## Task 11: predictor — 三组预测策略

**Files:**
- Modify: `predictor.py`
- Test: `tests/test_predictor.py` (追加)

- [ ] **Step 1: 写失败测试 — 三组预测生成**

追加到 `tests/test_predictor.py`（import 加 `from predictor import predict_group_a, predict_group_b, predict_group_c`，并补 `from data_fetcher import Record` 和 `from analysis import build_report`）:
```python
from data_fetcher import Record
from analysis import build_report

def mkrec(i):
    return Record(str(i), "t", [(i % 49) + 1]*6, (i*7 % 49) + 1)

class TestThreeGroups(unittest.TestCase):
    def setUp(self):
        random.seed(123)
        self.recs = [mkrec(i) for i in range(60)]
        self.report = build_report(self.recs)

    def test_group_a_valid(self):
        g = predict_group_a(self.recs, self.report)
        self.assertEqual(len(g.regular), 6)
        self.assertEqual(len(set(g.regular)), 6)
        self.assertIn("频率", g.strategy)

    def test_group_b_valid(self):
        g = predict_group_b(self.recs, self.report)
        self.assertEqual(len(set(g.regular)), 6)
        self.assertIn("马尔可夫", g.strategy)

    def test_group_c_without_llm(self):
        g = predict_group_c(self.recs, self.report, llm_result=None)
        self.assertEqual(len(set(g.regular)), 6)
        self.assertIn("遗漏", g.strategy)

    def test_group_c_with_llm(self):
        from llm_reasoner import LLMResult
        llm = LLMResult(inferred_models=["马尔可夫"], predicted_set=[3,12,18,25,33,41,7], reasoning="r")
        g = predict_group_c(self.recs, self.report, llm_result=llm)
        self.assertEqual(len(set(g.regular)), 6)

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试验证失败**

Run: `python -X utf8 -m unittest tests.test_predictor.TestThreeGroups -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: 实现三组预测**

追加到 `predictor.py`:
```python
def _freq_weights(report, boost_hot=True):
    """由频率构造权重; boost_hot=True 时热号权重更高。"""
    freq = report["freq"]
    base = {n: freq.get(n, 0) + 1.0 for n in range(1, 50)}  # +1 平滑
    if boost_hot:
        return base
    else:
        # 冷号回归: 遗漏越大权重越高
        gap = report["gap"]
        return {n: gap[n]["current"] + 1.0 for n in range(1, 50)}


def predict_group_a(records, report) -> PredictionGroup:
    """A 组: 频率/冷热加权, 生肖约束, 和值约束抽样。"""
    weights = _freq_weights(report, boost_hot=True)
    s = report["sum"]
    regular = sample_with_sum_constraint(weights, s["mean"], s["std"], k=6)
    # 特码取热号中最频
    special = max(range(1, 50), key=lambda n: report["freq"].get(n, 0))
    return PredictionGroup(name="A", regular=regular, special=special,
                           strategy="频率/冷热加权 + 和值约束")


def predict_group_b(records, report) -> PredictionGroup:
    """B 组: 马尔可夫特码转移 + 和值正态抽样。"""
    last_special = records[-1].special if records else 0
    trans = report["markov"].get(last_special, {})
    # 特码: 转移概率最高者
    if trans:
        special = max(trans, key=trans.get)
    else:
        special = random.randint(1, 49)
    weights = _freq_weights(report, boost_hot=True)
    s = report["sum"]
    regular = sample_with_sum_constraint(weights, s["mean"], s["std"], k=6)
    return PredictionGroup(name="B", regular=regular, special=special,
                           strategy="马尔可夫一阶转移 + 和值约束")


def predict_group_c(records, report, llm_result=None) -> PredictionGroup:
    """C 组: 大模型集成; 无 LLM 时退化为遗漏值回归。"""
    s = report["sum"]
    if llm_result and len(llm_result.predicted_set) >= 7:
        nums = [n for n in llm_result.predicted_set if 1 <= n <= 49]
        regular = nums[:6]
        special = nums[6] if len(nums) > 6 else nums[-1]
        # 若不足 6 个, 补齐
        while len(regular) < 6:
            cand = random.randint(1, 49)
            if cand not in regular:
                regular.append(cand)
        return PredictionGroup(name="C", regular=regular, special=special,
                               strategy="大模型集成推理")
    # 降级: 遗漏值回归(冷号即将出现)
    weights = _freq_weights(report, boost_hot=False)
    regular = sample_with_sum_constraint(weights, s["mean"], s["std"], k=6)
    special = max(range(1, 50), key=lambda n: report["gap"][n]["current"])
    return PredictionGroup(name="C", regular=regular, special=special,
                           strategy="遗漏值回归(无LLM降级)")
```

- [ ] **Step 4: 运行测试验证通过**

Run: `python -X utf8 -m unittest tests.test_predictor -v`
Expected: PASS (6 tests)

- [ ] **Step 5: 提交**

```bash
git add predictor.py tests/test_predictor.py
git commit -m "feat: predictor 三组预测策略"
```

---

## Task 12: predictor — 留一回测

**Files:**
- Modify: `predictor.py`
- Test: `tests/test_predictor.py` (追加)

- [ ] **Step 1: 写失败测试 — 回测命中率计算**

追加到 `tests/test_predictor.py`（import 加 `from predictor import backtest, predict_all`）:
```python
class TestBacktest(unittest.TestCase):
    def test_backtest_returns_rates(self):
        random.seed(7)
        recs = [mkrec(i) for i in range(60)]
        groups = predict_all(recs, llm_result=None, backtest_n=10)
        self.assertEqual(len(groups), 3)
        for g in groups:
            self.assertGreaterEqual(g.backtest_special_hit, 0.0)
            self.assertLessEqual(g.backtest_special_hit, 1.0)
            self.assertGreaterEqual(g.backtest_regular_hits, 0.0)

    def test_predict_all_sorts_by_hitrate(self):
        random.seed(7)
        recs = [mkrec(i) for i in range(60)]
        groups = predict_all(recs, llm_result=None, backtest_n=10)
        # 推荐组(第1个)命中率应 >= 其他
        rates = [g.backtest_special_hit + g.backtest_regular_hits / 10 for g in groups]
        self.assertGreaterEqual(rates[0], rates[-1])

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试验证失败**

Run: `python -X utf8 -m unittest tests.test_predictor.TestBacktest -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: 实现回测与 predict_all**

追加到 `predictor.py`:
```python
from analysis import build_report


def _evaluate(group: PredictionGroup, actual) -> dict:
    """评估单组预测对实际开奖的命中。actual: Record。"""
    reg_set = set(group.regular)
    actual_reg = set(actual.regular)
    reg_hits = len(reg_set & actual_reg)
    special_hit = 1 if group.special == actual.special else 0
    return {"reg_hits": reg_hits, "special_hit": special_hit}


def backtest(records, group_func, n: int = 30) -> tuple:
    """留一回测: 对最近 n 期, 每期用之前数据预测并评估。
    group_func(records, report) -> PredictionGroup。返回 (特码命中率, 平均平码命中数)。"""
    if len(records) < n + 20:
        n = max(1, len(records) - 20)
    special_hits = 0
    reg_hits_total = 0
    for i in range(n):
        split = len(records) - n + i
        train = records[:split]
        actual = records[split]
        report = build_report(train)
        group = group_func(train, report)
        ev = _evaluate(group, actual)
        special_hits += ev["special_hit"]
        reg_hits_total += ev["reg_hits"]
    return special_hits / n, reg_hits_total / n


def predict_all(records, llm_result=None, backtest_n: int = 30) -> list:
    """生成三组预测并回测, 按综合命中率降序排序。"""
    report = build_report(records)
    raw_groups = [
        predict_group_a(records, report),
        predict_group_b(records, report),
        predict_group_c(records, report, llm_result),
    ]
    funcs = [predict_group_a, predict_group_b,
             lambda r, rep: predict_group_c(r, rep, llm_result)]
    for g, fn in zip(raw_groups, funcs):
        sp, reg = backtest(records, fn, n=backtest_n)
        g.backtest_special_hit = sp
        g.backtest_regular_hits = reg
    # 综合分: 特码权重高
    raw_groups.sort(key=lambda g: (g.backtest_special_hit * 6 + g.backtest_regular_hits),
                    reverse=True)
    return raw_groups
```

- [ ] **Step 4: 运行测试验证通过**

Run: `python -X utf8 -m unittest tests.test_predictor -v`
Expected: PASS (8 tests)

- [ ] **Step 5: 提交**

```bash
git add predictor.py tests/test_predictor.py
git commit -m "feat: predictor 留一回测与综合排序"
```

---

## Task 13: macau_predictor — 主入口与 CLI 报告

**Files:**
- Create: `macau_predictor.py`
- Test: 手动运行验证

- [ ] **Step 1: 实现主入口**

`macau_predictor.py`:
```python
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
from predictor import predict_all

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

    print("[1/5] 加载历史数据...")
    records = load_history(years, refresh=args.refresh)
    if len(records) < 30:
        print("  ! 数据不足, 请检查网络或强制刷新(--refresh)。当前期数:", len(records))
        return
    print(f"  ✓ 共 {len(records)} 期, 时间范围 {records[0].open_time} ~ {records[-1].open_time}")
    print(f"  ✓ 最近一期: 期号 {records[-1].expect} 平码 {records[-1].regular} 特码 {records[-1].special}")
    print()

    print("[2/5] 统计分析与模型反推...")
    report = build_report(records)
    print("  反推理候选数学模型(拟合度 0-1, 越高越可能解释该序列):")
    for name, score, detail in report["models"]:
        print(f"    - {name}: {score:.3f}  ({detail})")
    print()

    print("[3/5] 大模型反推理(可选)...")
    cfg = load_config()
    llm_result = None
    if cfg:
        try:
            recent = [r.special for r in records[-10:]]
            summary = summarize_for_llm(report, recent_n=10)
            llm_result = reason(summary, recent, cfg)
            print("  ✓ 大模型反推理完成, 模型清单:", ", ".join(llm_result.inferred_models) or "(未解析)")
        except Exception as e:
            print(f"  ! 大模型调用失败, 降级为纯统计: {e}")
    else:
        print("  ! 未配置 LLM(编辑 config.ini 填入 api_key), 跳过大模型, 仅用统计层。")
    print()

    print("[4/5] 三组预测与回测...")
    groups = predict_all(records, llm_result=llm_result, backtest_n=args.backtest)
    for i, g in enumerate(groups):
        star = " ★推荐" if i == 0 else ""
        print(f"  {g.name}组{star}: 平码 {g.regular}  特码 {g.special}")
        print(f"      策略: {g.strategy}")
        print(f"      回测: 特码命中率={g.backtest_special_hit:.1%}  平均平码命中={g.backtest_regular_hits:.2f}/6")
    print()

    print("[5/5] 完成")
    print("-" * 70)
    print(DISCLAIMER)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 运行全部测试**

Run: `python -X utf8 -m unittest discover -s tests -v`
Expected: 全部 PASS

- [ ] **Step 3: 端到端运行(可离线, 用缓存; 首次需联网抓取)**

Run: `python -X utf8 macau_predictor.py`
Expected: 输出盒子标题、数据概览、模型清单、三组预测(标注特码)、回测命中率、推荐组、免责声明。无 LLM 配置时显示降级提示。

- [ ] **Step 4: 提交**

```bash
git add macau_predictor.py
git commit -m "feat: 主入口 CLI 报告与端到端流程"
```

---

## Task 14: 集成验证与文档

**Files:**
- Modify: `readme.md` (追加运行说明)

- [ ] **Step 1: 端到端冒烟运行**

Run: `python -X utf8 macau_predictor.py --backtest 20`
Expected: 正常输出, 退出码 0。若联网失败但本地有 `macau_history.json`, 仍能基于缓存运行。

- [ ] **Step 2: 更新 readme.md 运行说明**

在 `readme.md` 末尾追加:
```markdown

## 澳门六合彩预测软件 (macau_predictor)

运行方式:
  python -X utf8 macau_predictor.py [--refresh] [--years 2024,2025,2026] [--backtest 30]

大模型配置: 编辑 config.ini 填入 OpenAI 兼容接口的 base_url / api_key / model(如 DeepSeek)。
未配置时自动降级为纯统计模型。

输出: 反推理数学模型清单 + 三组预测(6平码+1特码) + 回测命中率 + 推荐组 + 科学免责声明。
```

- [ ] **Step 3: 全量测试最终确认**

Run: `python -X utf8 -m unittest discover -s tests -v`
Expected: 全部 PASS

- [ ] **Step 4: 提交**

```bash
git add readme.md
git commit -m "docs: 补充 macau_predictor 运行说明"
```

---

## 自检(对照 spec)

- 需求1(自动读取 macaujc 历史记录 + 反推理数学模型): Task 2-3 数据层 + Task 4-7 分析层 + Task 8-9 LLM 反推理 ✓
- 需求2(三组预测 + 命中率评估 + 哪组更高): Task 10-12 三组预测 + 留一回测 + 综合排序, Task 13 输出推荐组 ✓
- 需求3(反推理数学模型并罗列): Task 6 build_report 的 models 清单 + Task 9 LLM inferred_models + Task 13 输出 ✓
- 需求4(Python 实现): 全程 Python 标准库 ✓
- 科学免责声明: Task 13 DISCLAIMER ✓
- LLM 降级: Task 8 load_config + Task 9 reason 异常 + Task 11 group_c 降级 ✓
- 类型一致性: Record(expect/open_time/regular/special/waves/zodiacs)、AnalysisReport(build_report 返回 dict)、LLMResult(inferred_models/predicted_set/reasoning)、PredictionGroup(name/regular/special/strategy/backtest_*) 跨任务命名一致 ✓
