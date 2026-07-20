#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多维度预测层 — 针对特码的波色/生肖/尾数/大小/奇偶/头数预测"""
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass

from backtest_utils import backtest_series, rolling_stability

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        os.system("chcp 65001 >nul 2>&1")


@dataclass
class DimensionPrediction:
    name: str            # 维度名 e.g. "波色"
    value: str           # 预测值 e.g. "红波"
    strategy: str        # 策略描述
    accuracy: float = 0.0       # 回测准确率
    baseline: float = 0.0       # 随机基线
    lift: float = 0.0           # 提升度 = accuracy - baseline
    std_error: float = 0.0      # 标准误差
    stability: float = 0.0      # 滚动命中率标准差(越小越稳)
    mode_value: str = ""        # 全历史众数(参考, 不参与主预测)
    joint_number: int = 0       # 与六维预测最一致的参考号码(纯展示, 不反向决定维度)


@dataclass
class ZodiacPool:
    """特码生肖集合预测(单组, k 个生肖; k=3 或 4)。"""
    zodiacs: list        # k 个生肖
    strategy: str
    hit_rate: float = 0.0
    baseline: float = 0.0       # = 3/12
    lift: float = 0.0
    std_error: float = 0.0
    stability: float = 0.0


# ======================== 维度取值函数 ========================

def wave_of(rec) -> str:
    """特码波色(规范化为中文 红波/绿波/蓝波)。"""
    w = rec.waves[6] if len(rec.waves) > 6 else (rec.waves[-1] if rec.waves else "")
    return normalize_wave(w)


# 生肖简繁统一: 输出一律繁体(数据源偶发简体 龙/马/鸡/猪, 其余简繁同形)
_ZODIAC_TRAD = {"龙": "龍", "马": "馬", "鸡": "雞", "猪": "豬"}


def normalize_zodiac(z) -> str:
    """规范化生肖为繁体; 未知值原样返回, 空串保持空串。"""
    if not z:
        return ""
    s = str(z).strip()
    return _ZODIAC_TRAD.get(s, s)


def zodiac_of(rec) -> str:
    """特码生肖(统一繁体)。"""
    z = rec.zodiacs[6] if len(rec.zodiacs) > 6 else (rec.zodiacs[-1] if rec.zodiacs else "")
    return normalize_zodiac(z)


# 别名: 特码生肖(zodiac_of 的语义化名称)
def special_zodiac_of(rec) -> str:
    return zodiac_of(rec)


# 波色规范: 统一为中文 "红波"/"绿波"/"蓝波"
# (API 与历史缓存可能用英文 red/green/blue; 亦容忍裸 "红"/"绿"/"蓝")
_WAVE_ALIASES = {
    "red": "红波", "红波": "红波", "红": "红波",
    "green": "绿波", "绿波": "绿波", "绿": "绿波",
    "blue": "蓝波", "蓝波": "蓝波", "蓝": "蓝波",
}


def normalize_wave(w) -> str:
    """规范化波色为中文 红波/绿波/蓝波; 未知值原样返回, 空串保持空串。"""
    if not w:
        return ""
    s = str(w).strip()
    return _WAVE_ALIASES.get(s.lower(), s)


# 号码固定波色(标准六合彩 1-49; 经澳门 916 期数据验证 0 冲突)
_WAVE_OF_NUMBER = {
    1: "红波", 2: "红波", 7: "红波", 8: "红波", 12: "红波", 13: "红波", 18: "红波",
    19: "红波", 23: "红波", 24: "红波", 29: "红波", 30: "红波", 34: "红波", 35: "红波",
    40: "红波", 45: "红波", 46: "红波",
    3: "蓝波", 4: "蓝波", 9: "蓝波", 10: "蓝波", 14: "蓝波", 15: "蓝波", 20: "蓝波",
    25: "蓝波", 26: "蓝波", 31: "蓝波", 36: "蓝波", 37: "蓝波", 41: "蓝波", 42: "蓝波",
    47: "蓝波", 48: "蓝波",
    5: "绿波", 6: "绿波", 11: "绿波", 16: "绿波", 17: "绿波", 21: "绿波",
    22: "绿波", 27: "绿波", 28: "绿波", 32: "绿波", 33: "绿波", 38: "绿波",
    39: "绿波", 43: "绿波", 44: "绿波", 49: "绿波",
}


def wave_of_number(n: int) -> str:
    """号码(1-49)的固定波色 红波/蓝波/绿波; 越界或非法返回空串。"""
    try:
        return _WAVE_OF_NUMBER.get(int(n), "")
    except (TypeError, ValueError):
        return ""


def build_zodiac_map(records) -> dict:
    """由历史记录反推 号码->生肖 的当前映射(生肖统一繁体)。

    生肖按年轮换(非号码固定, 全量数据存在大量冲突), 故取每个号码"最近一次出现"
    时的生肖, 保证映射对应当前年度。records 需按时间升序。
    """
    m = {}
    for r in reversed(records):
        codes = list(r.regular) + [r.special]
        zs = r.zodiacs
        if len(zs) < len(codes):
            continue
        for n, z in zip(codes, zs):
            if n not in m and z:
                m[n] = normalize_zodiac(z)
    return m


def zodiac_of_number(n: int, zodiac_map: dict) -> str:
    """号码生肖(依当前年度映射); 无映射返回空串。"""
    return zodiac_map.get(int(n), "") if isinstance(n, int) else ""


def tail_of(rec) -> int:
    """尾数(个位 0-9)。"""
    return rec.special % 10


def big_small_of(rec) -> str:
    """大小: >=25 为大, 否则小。"""
    return "大" if rec.special >= 25 else "小"


def odd_even_of(rec) -> str:
    """奇偶。"""
    return "奇" if rec.special % 2 == 1 else "偶"


def head_of(rec) -> int:
    """头数(十位 0-4): 0-9->0, 10-19->1, ... 40-49->4。"""
    return rec.special // 10


# ======================== 三策略融合预测 ========================

def _freq_predict(train, extract):
    """频率策略: 历史众数(全期最频)。稳定但滞后, 仅作 mode_value 参考。"""
    c = Counter(extract(r) for r in train)
    if not c:
        return None
    return c.most_common(1)[0][0]


def _recent_freq_scores(train, extract, k=20, decay=0.92):
    """近期衰减频率分布: 取最近 k 期, 越近权重越大(decay^i), 返回 {值: 归一化得分}。
    归一化到 [0,1](除以最大), 供联合预测按维度加权求和。"""
    recent = train[-k:] if k > 0 else train
    if not recent:
        return {}
    scores = defaultdict(float)
    n = len(recent)
    for i, r in enumerate(recent):
        w = decay ** (n - 1 - i)  # 最近(i=n-1)权重最大
        v = extract(r)
        if v is not None:
            scores[v] += w
    if not scores:
        return {}
    mx = max(scores.values())
    if mx <= 0:
        return {}
    return {v: s / mx for v, s in scores.items()}


def _recent_freq_predict(train, extract, k=20, decay=0.92):
    """近期衰减频率: 返回加权频次最高的值(= _recent_freq_scores 的 argmax)。
    比全历史众数更随近期变动, 用于主预测值, 减少滞后。"""
    scores = _recent_freq_scores(train, extract, k, decay)
    if not scores:
        return None
    return max(scores.items(), key=lambda kv: kv[1])[0]


def _markov_predict(train, extract):
    """马尔可夫策略: 上一期值之后转移概率最高的值。"""
    if len(train) < 2:
        return None
    seq = [extract(r) for r in train]
    last = seq[-1]
    trans = defaultdict(Counter)
    for a, b in zip(seq[:-1], seq[1:]):
        trans[a][b] += 1
    if last in trans and trans[last]:
        return trans[last].most_common(1)[0][0]
    return None


def _trend_predict(train, extract, k=10):
    """近期趋势: 最近 k 期众数。"""
    recent = train[-k:]
    c = Counter(extract(r) for r in recent)
    if not c:
        return None
    return c.most_common(1)[0][0]


def _ensemble_predict(train, extract):
    """三策略融合(近期加权): 近期衰减频率/马尔可夫/趋势各投一票, 最多票者胜;
    平票取近期频率(随近期变动, 减少相对全历史众数的滞后)。

    全历史众数(_freq_predict)太稳、几乎总赢, 主预测值长期不变且滞后于走势反转;
    改用近期衰减频率作主投票, 使预测随近期数据变动。全历史众数另作 mode_value
    参考展示, 不参与主预测。
    """
    votes = Counter()
    for fn in (_recent_freq_predict, _markov_predict, _trend_predict):
        pred = fn(train, extract)
        if pred is not None:
            votes[pred] += 1
    if not votes:
        return _freq_predict(train, extract)
    # 平票时优先近期频率策略结果
    top = votes.most_common()
    max_count = top[0][1]
    winners = [v for v, c in top if c == max_count]
    rec_pred = _recent_freq_predict(train, extract)
    if rec_pred in winners:
        return rec_pred
    return winners[0]


# 维度配置: (名称, 取值函数)。基线不再写死——改为 comb_prior 的精确组合概率。
DIMENSIONS = [
    ("波色", wave_of),
    ("生肖", zodiac_of),
    ("尾数", tail_of),
    ("大小", big_small_of),
    ("奇偶", odd_even_of),
    ("头数", head_of),
]


def comb_prior(name, zodiac_map=None) -> dict:
    """各维度取值的精确组合先验(特码 1-49 均匀抽取下的真实概率)。

    波色 红17/蓝16/绿16; 尾数 0→4个号码, 1-9→5个; 大小 大25/小24;
    奇偶 奇25/偶24; 头数 0→9个, 1-4→10个; 生肖按当前年度映射的号码数(胖5/瘦4)。
    生肖在映射为空时返回 {}。这是唯一诚实的基线: 1/3、1/10、1/5 等近似全是错的。
    """
    if name == "波色":
        return {"红波": 17 / 49, "蓝波": 16 / 49, "绿波": 16 / 49}
    if name == "生肖":
        if not zodiac_map:
            return {}
        # 映射未满 49 个号码时先验不归一(为下界, 保守); 满 49 时精确
        cnt = Counter(zodiac_map.values())
        return {z: c / 49 for z, c in cnt.items()}
    if name == "尾数":
        return {t: (4 if t == 0 else 5) / 49 for t in range(10)}
    if name == "大小":
        return {"大": 25 / 49, "小": 24 / 49}
    if name == "奇偶":
        return {"奇": 25 / 49, "偶": 24 / 49}
    if name == "头数":
        return {h: (9 if h == 0 else 10) / 49 for h in range(5)}
    return {}


# 每维预测方式: None = 纯先验 argmax(静态); (S, W) = 后验 argmax(先验×S + 近 W 期计数)。
#
# 判据(walk-forward 932 期验证, 选择窗/报告窗分离): 彩票近随机, 任何动态信号样本外
# 都回落到基线——所以只在"先验平局区"允许翻动, 那里翻动的期望成本≈0:
#   尾数(尾1-9 同为 5/49): 先验完全相等, 由近期走势决定, 两窗实测不劣于静态;
#   大小/奇偶(25:24, 差 1/49): 先验近等, 强近期证据才翻案, 实测两窗为正;
#   波色(17:16:16)、生肖(5:4): 先验差距实质, 翻动=牺牲必然优势, 取纯先验 argmax;
#   头数(头1-4 同为 10/49): 先验虽等, 但动态变体近 500 期实测全部为负(计数噪声
#   把先验略低的头0(9/49)也顶上来), 期望无收益, 取纯先验 argmax(确定性取头1)。
# 静态维度的期望命中=精确基线, 这是近随机假设下数学上可证的最优。
DIM_CONFIG = {
    "波色": None,
    "生肖": None,
    "尾数": (40.0, 40),
    "大小": (40.0, 40),
    "奇偶": (40.0, 40),
    "头数": None,
}

_DIM_STRATEGY = {
    "波色": "精确组合先验最优(红17/49 > 蓝绿16/49), 无可靠信号时恒选红=理论最优",
    "生肖": "当前年度胖生肖(5个号码, 5/49 > 瘦4/49); 映射由全历史最近出现推断, 随年度轮换自动切换",
    "尾数": "贝叶斯后验(精确组合先验×40 + 近40期计数); 尾1-9先验相等, 由近期走势决定",
    "大小": "贝叶斯后验(先验 大25/小24 ×40 + 近40期计数); 先验近等, 强近期证据可翻案",
    "奇偶": "贝叶斯后验(先验 奇25/偶24 ×40 + 近40期计数); 先验近等, 强近期证据可翻案",
    "头数": "精确组合先验最优(头1-4 各10/49 > 头0 9/49); 近期信号回测无持续优势, 取静态最大类",
}


def _posterior_scores(records, extract, prior, cfg) -> dict:
    """各候选值得分(归一化到 [0,1])。
    cfg=None: 纯先验(静态最优); cfg=(S,W): 后验 = prior*S + 近 W 期计数。
    只统计 prior 键内的取值, 键外忽略。"""
    if not prior:
        return {}
    if cfg is None:
        scores = dict(prior)
    else:
        s_strength, window = cfg
        cnt = Counter(extract(r) for r in records[-window:])
        scores = {v: p * s_strength + cnt.get(v, 0) for v, p in prior.items()}
    mx = max(scores.values()) if scores else 0.0
    if mx <= 0:
        return {}
    return {v: s / mx for v, s in scores.items()}


def _dim_predict_value(records, extract, prior, cfg):
    """单维预测值: 后验(或纯先验)argmax; prior 为空回退全历史众数。"""
    scores = _posterior_scores(records, extract, prior, cfg)
    if scores:
        return max(scores.items(), key=lambda kv: kv[1])[0]
    return _freq_predict(records, extract)


def _dim_value_of_number(n, name, zodiac_map):
    """号码 n 在指定维度上的取值(与 Record 取值函数同口径)。"""
    if name == "波色":
        return wave_of_number(n)
    if name == "生肖":
        return zodiac_of_number(n, zodiac_map)
    if name == "尾数":
        return n % 10
    if name == "大小":
        return "大" if n >= 25 else "小"
    if name == "奇偶":
        return "奇" if n % 2 == 1 else "偶"
    if name == "头数":
        return n // 10
    return None


# 反持续性: 对"上一期实际特码"的参考号得分乘此系数(仅影响展示用参考号)。
# 1.0=关闭, 0.0=等同禁绝, 默认半折。旧版六维由单一号码导出时, 上一期特码被
# 最近一期最高权重同时顶高导致"预测=上一期"; 现六维独立后该伪影已消除,
# 保留折扣仅为参考号观感(避免参考号连续重复上一期)。
_LAST_SPECIAL_DISCOUNT = 0.5


def _joint_predict_number(records, zodiac_map, discount_last=_LAST_SPECIAL_DISCOUNT):
    """与六维预测最一致的参考号码(纯展示): 按各维后验得分求和 argmax。

    六维主预测各自独立(见 predict_dimensions), 本号码不再反向决定维度值;
    它随各维预测翻动而翻动。候选仅限 zodiac_map 中已有生肖映射的号码,
    映射空时回退全 1-49。discount_last: 上一期特码得分乘此系数。
    """
    zm = zodiac_map or {}
    dim_scores = {name: _posterior_scores(records, ex, comb_prior(name, zm), DIM_CONFIG.get(name))
                  for name, ex in DIMENSIONS}
    candidates = [n for n in range(1, 50) if n in zm] or list(range(1, 50))
    try:
        last_special = int(records[-1].special) if records else None
    except (TypeError, ValueError):
        last_special = None
    best_n, best_s = candidates[0], -1.0
    for n in candidates:  # 升序遍历 + 严格大于: 同分时取号码最小者(确定性, 便于复现)
        s = 0.0
        for name, _ex in DIMENSIONS:
            v = _dim_value_of_number(n, name, zm)
            s += dim_scores.get(name, {}).get(v, 0.0)
        if n == last_special:
            s *= discount_last  # 反持续性: 软惩罚上一期特码
        if s > best_s:
            best_s = s
            best_n = n
    return best_n


def predict_dimensions(records, llm_result=None, backtest_n: int = 30) -> list:
    """六维独立贝叶斯预测 + 精确组合基线 + 留一回测。

    每维预测 = comb_prior 精确先验的后验 argmax(先验平局区由近期计数翻动,
    先验有实质差距的维度取纯先验, 详见 DIM_CONFIG)。彩票近随机: 本预测的期望
    命中≈该维精确基线, 这是数学上可证的最优; 回测 lift 在 0 附近波动属正常,
    持续大幅为正反而说明回测有前视。
    基线口径: 回测窗口内"当时预测值"的精确组合概率均值(与预测同口径, 无前视)。
    联合参考号(joint_number)仅展示, 为与六维预测最一致的号码, 不反向决定维度。
    mode_value 为各维度独立的全历史众数(参考)。
    """
    import math
    zodiac_map = build_zodiac_map(records) if records else {}
    joint_n = _joint_predict_number(records, zodiac_map)

    # 留一回测: 每期仅用之前数据, 各维独立预测并与实际比对; 同口径记录当时预测值的先验。
    # 先验为空的折(仅生肖映射在数据极短时可能为空)整折跳过, 保证 dim_base 与 dim_hits
    # 等长、baseline 与 accuracy 同口径。
    n_clamped = max(1, min(backtest_n, len(records) - 10))
    dim_hits = {name: [] for name, _ex in DIMENSIONS}
    dim_base = {name: [] for name, _ex in DIMENSIONS}
    for i in range(n_clamped):
        split = len(records) - n_clamped + i
        train = records[:split]
        actual = records[split]
        train_zmap = build_zodiac_map(train)
        for name, extract in DIMENSIONS:
            prior = comb_prior(name, train_zmap)
            if not prior:
                continue
            pred = _dim_predict_value(train, extract, prior, DIM_CONFIG.get(name))
            dim_hits[name].append(1 if pred is not None and pred == extract(actual) else 0)
            dim_base[name].append(prior[pred] if pred in prior else 1.0 / len(prior))

    results = []
    for name, extract in DIMENSIONS:
        hits = dim_hits[name]
        bases = dim_base[name]
        acc = sum(hits) / len(hits) if hits else 0.0
        baseline = sum(bases) / len(bases) if bases else 0.0
        se = math.sqrt(acc * (1 - acc) / len(hits)) if hits else 0.0
        stab = rolling_stability(hits)
        prior_now = comb_prior(name, zodiac_map)
        value = _dim_predict_value(records, extract, prior_now, DIM_CONFIG.get(name))
        mode = _freq_predict(records, extract)
        results.append(DimensionPrediction(
            name=name,
            value=str(value) if value is not None else "",
            mode_value=str(mode) if mode is not None else "",
            joint_number=joint_n,
            strategy=_DIM_STRATEGY.get(name, ""),
            accuracy=acc, baseline=baseline, lift=acc - baseline,
            std_error=se, stability=stab,
        ))
    return results


# ======================== 特码生肖集合预测(单组) ========================

ZODIAC_POOL_SIZE = 3
ZODIAC_QUAD_SIZE = 4
ZODIAC_SIX_SIZE = 6


def _build_zodiac_pool(records, k=ZODIAC_POOL_SIZE, zodiac_map=None, llm_result=None):
    """按融合得分取前 k 个生肖(与六肖 _zodiac_six_scores 同口径, llm_weight=0 纯统计)。

    统一打分后 top3⊆top4 成立: 三/四肖取同一 scores 的前 k 个切片。
    (⊆top6 仅在六肖也 llm_weight=0 时成立; 六肖开 LLM 权重后打分口径不同,
    包含关系不再保证——见 predict_zodiac_six。)
    zodiac_map 未传时就地按 records 构建, 保证贝叶斯/号码数信号与六肖一致不退化。
    """
    if zodiac_map is None:
        zodiac_map = build_zodiac_map(records) if records else {}
    scores = _zodiac_six_scores(records, llm_result, zodiac_map)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    zodiacs = [z for z, _ in ranked[:k]]
    # 不足 k 个补齐(按频率)
    if len(zodiacs) < k:
        seq = [special_zodiac_of(r) for r in records if special_zodiac_of(r)]
        c = Counter(seq)
        for z, _ in c.most_common():
            if z not in zodiacs:
                zodiacs.append(z)
                if len(zodiacs) == k:
                    break
    return zodiacs[:k]


def predict_zodiac_pool(records, llm_result=None, backtest_n: int = 30,
                        k: int = ZODIAC_POOL_SIZE) -> ZodiacPool:
    """生成单组 k 个最可能开出特码的生肖, 并回测。

    与六肖共用 _zodiac_six_scores 统一打分(llm_weight=0, 纯统计), 保证
    top3⊆top4⊆top6。回测每折用 train 自建 zodiac_map, 不传 llm_result(无未来泄漏)。
    """
    import math
    baseline = k / 12
    zodiac_map = build_zodiac_map(records) if records else {}
    zodiacs = _build_zodiac_pool(records, k, zodiac_map=zodiac_map, llm_result=llm_result)

    # 留一回测: 真实特码生肖落在 k 个内即命中
    n_clamped = max(1, min(backtest_n, len(records) - 10))
    hits = []
    for i in range(n_clamped):
        split = len(records) - n_clamped + i
        train = records[:split]
        actual = records[split]
        pool = set(_build_zodiac_pool(train, k, zodiac_map=build_zodiac_map(train)))
        hits.append(1 if special_zodiac_of(actual) in pool else 0)
    accuracy = sum(hits) / n_clamped if n_clamped else 0.0
    std_error = math.sqrt(accuracy * (1 - accuracy) / n_clamped) if n_clamped else 0.0
    stab = rolling_stability(hits)

    return ZodiacPool(
        zodiacs=zodiacs,
        strategy="频率+马尔可夫backoff+跨维度融合+近期热度+贝叶斯+号码数 多信号融合(与六肖同口径)",
        hit_rate=accuracy, baseline=baseline, lift=accuracy - baseline,
        std_error=std_error, stability=stab,
    )


def predict_zodiac_quad(records, llm_result=None, backtest_n: int = 30) -> ZodiacPool:
    """生成单组 4 个最可能开出特码的生肖, 并回测。"""
    return predict_zodiac_pool(records, llm_result=llm_result,
                               backtest_n=backtest_n, k=ZODIAC_QUAD_SIZE)


class _SigCtx:
    """信号上下文: 各 _signal_xxx 共享的预计算数据, 避免重复扫描序列。"""
    __slots__ = ("seq", "all_zodiacs", "llm_result", "zodiac_map", "llm_weight", "records")

    def __init__(self, seq, all_zodiacs, llm_result, zodiac_map, llm_weight=0.0, records=None):
        self.seq = seq
        self.all_zodiacs = all_zodiacs
        self.llm_result = llm_result
        self.zodiac_map = zodiac_map
        # LLM 无法回测(历史无 LLM 预测可重放): 回测恒为 0, 预测可由调用方开启。
        # _signal_llm 据此决定是否贡献, 保证上报命中率与预测口径一致。
        self.llm_weight = llm_weight
        # 原始 records(含平码/波色等), 供跨维度信号(_signal_cross_dim)复用, 避免重复透传。
        self.records = records


# ======================== 六肖打分信号(各自独立, 便于单独优化) ========================
# 每个信号函数签名: (ctx: _SigCtx) -> dict[zodiac -> 贡献分]
# 权重内嵌在各信号内; _zodiac_six_scores 按 _ZODIAC_SIX_SIGNALS 顺序累加。
# 约定: 仅返回正贡献; 落到不在 all_zodiacs 中的生肖由调用方过滤(等价于原 if z in scores)。

def _signal_freq(ctx):
    """频率(归一化) - 权重 3.0。历史出现最多的生肖得分最高。"""
    cnt = Counter(ctx.seq)
    max_cnt = max(cnt.values()) if cnt else 1
    return {z: (c / max_cnt) * 3.0 for z, c in cnt.items()}


def _signal_markov(ctx):
    """马尔可夫插值 backoff - 权重 4.0。
    P(z) = λ2·P2(z|上两期) + λ1·P1(z|上期) + λ0·P0(z)(unigram); λ 固定。
    二阶状态无观测时 P2 项为 0(等效该项降权, 权重不重归一化——稀疏上下文被
    系统性压低, 属保守设计而非"自动退化")。λ2=0.5, λ1=0.3, λ0=0.2。
    归一化到 [0,1] 后乘权重 4.0(原1阶2.5+2阶2.0=4.5, backoff更平滑故略降)。
    """
    seq = ctx.seq
    if len(seq) < 2:
        return {}
    cnt_all = Counter(seq)
    total = len(seq)
    trans1 = defaultdict(Counter)
    for a, b in zip(seq[:-1], seq[1:]):
        trans1[a][b] += 1
    last1 = seq[-1]
    row1 = trans1.get(last1, Counter())
    s1 = sum(row1.values())
    row2 = Counter()
    s2 = 0
    if len(seq) >= 3:
        trans2 = defaultdict(Counter)
        for a, b, c in zip(seq[:-2], seq[1:-1], seq[2:]):
            trans2[(a, b)][c] += 1
        last2 = (seq[-2], seq[-1])
        row2 = trans2.get(last2, Counter())
        s2 = sum(row2.values())
    lam2, lam1, lam0 = 0.5, 0.3, 0.2
    out = {}
    for z in cnt_all:
        p2 = (row2[z] / s2) if s2 > 0 else 0.0
        p1 = (row1[z] / s1) if s1 > 0 else 0.0
        p0 = cnt_all[z] / total
        out[z] = lam2 * p2 + lam1 * p1 + lam0 * p0
    omax = max(out.values()) if out else 1.0
    if omax <= 0:
        return {}
    return {z: (v / omax) * 4.0 for z, v in out.items()}


def _signal_recent(ctx):
    """近期热度(指数衰减加权频率, 窗口 12 期) - 权重 2.8。

    合并原 trend(近10期均权)+ wma(近5期线性加权): 二者高度相关, 合并为单一指数衰减
    加权频率, 越近期次权重越大(λ=0.82), 归一化后乘权重。窗口 12≈一个生肖轮回,
    兼顾短期热度与去噪。合并权重 2.8(原 trend 2.0 + wma 1.5 = 3.5, 衰减更平滑故略降)。
    """
    window = ctx.seq[-12:]
    if not window:
        return {}
    lam = 0.82
    weights = [lam ** (len(window) - 1 - i) for i in range(len(window))]
    recent = defaultdict(float)
    for w, z in zip(weights, window):
        recent[z] += w
    r_max = max(recent.values()) if recent else 1.0
    if r_max <= 0:
        return {}
    return {z: (v / r_max) * 2.8 for z, v in recent.items()}


def _signal_bayes(ctx):
    """贝叶斯后验概率 - 权重 1.5。
    先验用胖瘦生肖 informative prior(5号码生肖 5/49, 4号码生肖 4/49, 唯一有理论依据的先验),
    无 zodiac_map 时回退均匀先验; 似然为近 15 期频率(拉普拉斯平滑);
    后验归一化到 [0,1] 再乘权重, 与其它信号量纲可比。
    """
    seq = ctx.seq
    if len(seq) < 10 or not ctx.all_zodiacs:
        return {}
    all_zodiacs = ctx.all_zodiacs
    if ctx.zodiac_map:
        num_count = Counter(ctx.zodiac_map.values())
        prior = {z: num_count.get(z, 0) / 49.0 for z in all_zodiacs}
    else:
        uniform = 1.0 / len(all_zodiacs)
        prior = {z: uniform for z in all_zodiacs}
    recent15 = seq[-15:]
    cnt_likely = Counter(recent15)
    total_likely = len(recent15)
    K = len(all_zodiacs)
    raw = {}
    for z in all_zodiacs:
        likelihood = (cnt_likely.get(z, 0) + 1) / (total_likely + K)
        raw[z] = likelihood * prior[z]
    rmax = max(raw.values()) if raw else 1.0
    if rmax <= 0:
        return {}
    return {z: (v / rmax) * 1.5 for z, v in raw.items()}


def _signal_llm(ctx):
    """LLM 反推理信号 - 基础权重 2.0, 受 ctx.llm_weight 调节。

    LLM 无法回测(历史期次没有可重放的 LLM 预测), 故回测时 llm_weight 恒为 0,
    本信号完全不贡献, 保证上报命中率与纯统计预测口径一致;
    预测时调用方可令 llm_weight>0 让 LLM 参与选号(默认 0 = 关闭)。
    """
    if not ctx.llm_weight:
        return {}
    if not (ctx.llm_result and ctx.llm_result.predicted_set and ctx.zodiac_map):
        return {}
    llm_zodiac_cnt = Counter()
    for num in ctx.llm_result.predicted_set:
        z = ctx.zodiac_map.get(num, "")
        if z:
            llm_zodiac_cnt[z] += 1
    if not llm_zodiac_cnt:
        return {}
    llm_max = max(llm_zodiac_cnt.values())
    return {z: (c / llm_max) * 2.0 * ctx.llm_weight for z, c in llm_zodiac_cnt.items()}


def _signal_numcount(ctx):
    """号码数加权 - 权重 1.5。胖生肖(5 个号码)概率高于瘦生肖(4 个号码)。"""
    if not ctx.zodiac_map:
        return {}
    num_count = Counter(ctx.zodiac_map.values())
    nc_max = max(num_count.values()) if num_count else 1
    return {z: (cnt / nc_max) * 1.5 for z, cnt in num_count.items()}


def _signal_cross_dim(ctx):
    """跨维度条件融合 - 权重 2.0。
    用三策略融合(_ensemble_predict)预测下一期特码的波色/头数/尾数,
    按各生肖号码与预测值的匹配占比加分(多维度求和归一化)。
    需要 ctx.records; 未提供时安全返回 {}(集成阶段才接入)。
    """
    records = getattr(ctx, "records", None)
    if not records or not ctx.zodiac_map or not ctx.all_zodiacs:
        return {}
    try:
        pred_wave = normalize_wave(_ensemble_predict(records, wave_of) or "")
        pred_head = _ensemble_predict(records, head_of)
        pred_tail = _ensemble_predict(records, tail_of)
    except Exception:
        return {}
    zod_to_nums = defaultdict(list)
    for num, z in ctx.zodiac_map.items():
        zod_to_nums[z].append(num)
    out = {}
    for z in ctx.all_zodiacs:
        nums = zod_to_nums.get(z, [])
        if not nums:
            continue
        n = len(nums)
        score = 0.0
        if pred_wave:
            score += sum(1 for x in nums if wave_of_number(x) == pred_wave) / n
        if pred_head is not None:
            score += sum(1 for x in nums if (x // 10) == pred_head) / n
        if pred_tail is not None:
            score += sum(1 for x in nums if (x % 10) == pred_tail) / n
        out[z] = score
    if not out:
        return {}
    omax = max(out.values()) if out else 1.0
    if omax <= 0:
        return {}
    return {z: (v / omax) * 2.0 for z, v in out.items()}


# 六肖信号注册表: (名称, 函数)。顺序即累加顺序; 增删/重排在此处操作。
_ZODIAC_SIX_SIGNALS = [
    ("freq", _signal_freq),
    ("markov", _signal_markov),       # B2: backoff 插值, 取代 markov1/markov2
    ("cross_dim", _signal_cross_dim), # B3: 跨维度(波色/头数/尾数)条件融合
    ("recent", _signal_recent),       # B5: 指数衰减近期热度, 取代 trend/wma
    ("bayes", _signal_bayes),
    ("llm", _signal_llm),
    ("numcount", _signal_numcount),
]


def _zodiac_six_scores(records, llm_result=None, zodiac_map=None, llm_weight=0.0):
    """融合多种算法为每个生肖打分: 频率、马尔可夫(backoff)、跨维度融合、
    近期热度、贝叶斯后验、LLM信号、号码数加权。

    各信号实现见 _signal_xxx; 通过 _ZODIAC_SIX_SIGNALS 注册表按序累加,
    仅累加到出现在历史序列(all_zodiacs)中的生肖, 其余丢弃。
    records 透传给 ctx 供跨维度信号(_signal_cross_dim)复用。

    llm_weight: LLM 信号开关(回测恒 0, 预测可开), 详见 _signal_llm。
    """
    seq = [special_zodiac_of(r) for r in records if special_zodiac_of(r)]
    if not seq:
        return {}
    # sorted: 确定性顺序(平分时的兜底排序可复现, 不受 PYTHONHASHSEED 影响)
    all_zodiacs = sorted(set(seq))
    scores = {z: 0.0 for z in all_zodiacs}
    ctx = _SigCtx(seq, all_zodiacs, llm_result, zodiac_map, llm_weight, records)
    for _name, fn in _ZODIAC_SIX_SIGNALS:
        contrib = fn(ctx)
        if not contrib:
            continue
        for z, v in contrib.items():
            if z in scores:
                scores[z] += v
    return scores


def _build_zodiac_six(records, llm_result=None, zodiac_map=None, llm_weight=0.0):
    """按融合得分取前 6 个生肖(与三/四肖同口径 _zodiac_six_scores)。

    llm_weight 透传给 _zodiac_six_scores: 预测时可开, 回测恒 0(不泄漏未来)。
    注: top3⊆top4⊆top6 仅在 llm_weight=0 时成立(开 LLM 权重后六肖打分口径不同)。
    """
    scores = _zodiac_six_scores(records, llm_result, zodiac_map, llm_weight)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    zodiacs = [z for z, _ in ranked[:ZODIAC_SIX_SIZE]]
    # 不足 6 个(数据极稀)按历史频率补足
    if len(zodiacs) < ZODIAC_SIX_SIZE:
        seq = [special_zodiac_of(r) for r in records if special_zodiac_of(r)]
        cnt = Counter(seq)
        for z, _ in cnt.most_common():
            if z not in zodiacs:
                zodiacs.append(z)
                if len(zodiacs) == ZODIAC_SIX_SIZE:
                    break
    return zodiacs[:ZODIAC_SIX_SIZE]


def _llm_reference_zodiacs(llm_result, zodiac_map, topn=2):
    """把 LLM 预测号码经 zodiac_map 映射到生肖, 取出现次数最多的 topn 个, 用作文案展示。

    LLM 无法回测, 此结果仅追加到 strategy 末尾作"大模型参考"展示,
    不参与选号打分(选号是否用 LLM 由 llm_weight 控制, 见 _signal_llm)。
    返回 "鼠、龙" 形式字符串; 无可用映射时返回 ""。
    """
    if not (llm_result and getattr(llm_result, "predicted_set", None) and zodiac_map):
        return ""
    cnt = Counter()
    for num in llm_result.predicted_set:
        z = zodiac_map.get(num, "")
        if z:
            cnt[z] += 1
    if not cnt:
        return ""
    return "、".join(z for z, _ in cnt.most_common(topn))


def predict_zodiac_six(records, llm_result=None, backtest_n: int = 30,
                       llm_weight: float = 0.0) -> ZodiacPool:
    """生成单组 6 个最可能开出特码的生肖, 并回测。

    算法: 频率 + 马尔可夫backoff + 跨维度融合 + 近期热度 + 贝叶斯后验 + LLM反推理信号 融合。

    llm_weight: LLM 信号开关。LLM 无法回测(历史无 LLM 预测可重放), 故回测恒用 0,
    保证上报命中率与纯统计口径一致; 预测默认亦为 0, 调用方可令其>0 让 LLM 参与选号。
    无论是否参与选号, 只要传入 llm_result 即在 strategy 末尾附"大模型参考"展示。
    """
    import math
    baseline = ZODIAC_SIX_SIZE / 12

    zodiac_map = build_zodiac_map(records) if records else {}
    zodiacs = _build_zodiac_six(records, llm_result, zodiac_map, llm_weight)

    # 留一回测: 每期仅用之前数据预测，真实特码生肖落在 6 个内即命中
    # 回测恒用 llm_weight=0(且 llm_result=None): LLM 无法回测, 不泄漏未来, 命中率口径与预测一致
    n_clamped = max(1, min(backtest_n, len(records) - 10))
    hits = []
    for i in range(n_clamped):
        split = len(records) - n_clamped + i
        train = records[:split]
        actual = records[split]
        train_zmap = build_zodiac_map(train)
        pred = _build_zodiac_six(train, None, train_zmap, 0.0)
        actual_z = special_zodiac_of(actual)
        hits.append(1 if actual_z in set(pred) else 0)

    hit_rate = sum(hits) / n_clamped if n_clamped else 0.0
    std_error = math.sqrt(hit_rate * (1 - hit_rate) / n_clamped) if n_clamped else 0.0
    stability = rolling_stability(hits)

    strategy = "频率+马尔可夫backoff+跨维度融合+近期热度+贝叶斯后验+号码数加权 多算法融合"
    if llm_weight:
        strategy += "+LLM信号"
    llm_ref = _llm_reference_zodiacs(llm_result, zodiac_map)
    if llm_ref:
        strategy += f"（大模型参考: {llm_ref}）"

    return ZodiacPool(
        zodiacs=zodiacs,
        strategy=strategy,
        hit_rate=hit_rate,
        baseline=baseline,
        lift=hit_rate - baseline,
        std_error=std_error,
        stability=stability,
    )
