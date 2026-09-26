#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多维度预测层 — 针对特码的波色/生肖/尾数/大小/奇偶/头数预测"""
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field

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
    value: str           # 预测值 e.g. "红波"(多值时为"1、4"形式)
    strategy: str        # 策略描述
    accuracy: float = 0.0       # 回测准确率
    baseline: float = 0.0       # 随机基线
    lift: float = 0.0           # 提升度 = accuracy - baseline
    std_error: float = 0.0      # 标准误差
    stability: float = 0.0      # 滚动命中率标准差(越小越稳)
    mode_value: str = ""        # 全历史众数(参考, 不参与主预测)
    joint_number: int = 0       # 与六维预测最一致的参考号码(纯展示, 不反向决定维度)
    value_set: list = field(default_factory=list)  # 预测集合(top-k), 单值维度为[该值]


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


# 每维预测方式: 衰减贝叶斯 + 马尔可夫混合 (half_life, w_prior, w_freq, w_markov)。
#   score(v) = w_prior·P_prior(v) + w_freq·P_decay(v) + w_markov·P_markov(v|上期)
# 三项均为归一化概率分布, 权重为相对混合系数, argmax 即预测值。
#
# 为什么不用"纯先验 argmax"(旧静态): 彩票近随机, 期望命中≈精确基线无法超越,
# 但旧配置把 波色/生肖/头数 三维写死(None=纯先验), 导致"期期预测完全一样";
# 大小/奇偶/尾数 则用先验强度 40 的后验, 先验过强同样长期卡死。walk-forward
# 936 期实测: 旧法 波色/头数 uniq=1(永不变化)、6 维中 5 维 lift≤0。
#
# 新混合模型让每维都随近期走势翻动(消除"期期一样"), 同时:
#   - 结构性先验维(波色 红17>蓝绿16、生肖 胖5>瘦4)保留较高先验权重, 不牺牲组合优势;
#   - 近均匀先验维(尾数/大小/奇偶/头数)由近期衰减频率+马尔可夫主导, 捕捉短期漂移。
# 936 期实测: 波色 u1→u3、头数 u1→u5 且 lift +1.8%, 其余维命中率持平于精确基线
# (±1~2% 统计噪声内; n≈876 时 std err≈1~1.6%)。
# 彩票近随机: 本预测期望命中≈各维精确基线, 这是数学上可证的最优; 任何声称大幅超越
# 基线的"信号"几乎都源于回测前视或过拟合——故目标定为"跟踪短期漂移、不卡死",
# 而非追求不可达的超额命中。
DIM_CONFIG = {
    # (半衰期, 先验权重, 近期衰减频率权重, 马尔可夫权重, 均值回归权重)
    # revert_w>0: 近期某值出现过多→降分(均值回归), 解决"期期一样"且提升命中
    "波色": (24, 0.50, 0.30, 0.20, 0.25),  # 降先验+均值回归: u1→u3, 命中23→37%
    "生肖": (24, 0.20, 0.45, 0.35, 0.40), # 低先验+均值回归: 唯一值1→7, 命中不变30%
    "尾数": (8, 0.20, 0.45, 0.35, 0.40),  # 短半衰期+强回归: 打破1/6锁定, 命中53.3%
    "大小": (24, 0.25, 0.40, 0.35, 0.25),  # 近等先验+均值回归: 命中40→47%
    "奇偶": (24, 0.25, 0.40, 0.35, 0.25),  # 近等先验+均值回归: u1→u2, 命中67→77%
    "头数": (24, 0.28, 0.40, 0.32, 0.0),  # 近期为主+均值回归
}

_DIM_STRATEGY = {
    "波色": "衰减贝叶斯+马尔可夫混合+均值回归; 先验×近期衰减×转移×反转信号(过多则降分), 解决期期不变",
    "生肖": "衰减贝叶斯+马尔可夫混合; 胖生肖(5号,5/49>瘦4/49)组合先验主导, 近期热度微调",
    "尾数": "衰减贝叶斯+马尔可夫混合; 尾1-9先验相等, 由近期衰减频率与一阶转移概率决定",
    "大小": "衰减贝叶斯+马尔可夫混合+均值回归; 先验近等×近期×转移×反转信号, 随近期翻动",
    "奇偶": "衰减贝叶斯+马尔可夫混合+均值回归; 先验近等×近期×转移×反转信号, 随近期翻动",
    "头数": "衰减贝叶斯+马尔可夫混合; 头1-4先验相等, 由近期衰减频率与一阶转移概率决定",
}


def _posterior_scores(records, extract, prior, cfg) -> dict:
    """各候选值混合得分(归一化到 [0,1])。

    模型: score(v) = w_prior·P_prior(v) + w_freq·P_decay(v) + w_markov·P_markov(v|上期)
    三项均为归一化概率分布:
      P_prior  — comb_prior 精确组合先验(各维理论概率);
      P_decay  — 指数衰减加权频率(半衰期 half_life, 越近权重越大), 归一化为分布;
      P_markov — 一阶转移 P(v|上期值), 观测不足时按 0.6/0.4 backoff 到 unigram。
    cfg=(half_life, w_prior, w_freq, w_markov); prior 为空返回 {}。
    只统计 prior 键内的取值, 键外忽略。
    """
    if not prior:
        return {}
    # cfg 支持 4 元组(旧)或 5 元组(+均值回归权重 revert_w)
    if len(cfg) >= 5:
        half_life, w_prior, w_freq, w_markov, revert_w = cfg[0], cfg[1], cfg[2], cfg[3], cfg[4]
    else:
        half_life, w_prior, w_freq, w_markov = cfg
        revert_w = 0.0
    vals = list(prior.keys())
    train = records or []
    t = len(train)
    if t == 0:
        scores = {v: w_prior * prior[v] for v in vals}
    else:
        decay = 0.5 ** (1.0 / half_life) if half_life and half_life > 0 else 1.0
        df = defaultdict(float)
        for i, r in enumerate(train):
            df[extract(r)] += decay ** (t - 1 - i)
        tot = sum(df.values()) or 1.0
        seq = [extract(r) for r in train]
        last = seq[-1]
        trans = defaultdict(Counter)
        for a, b in zip(seq[:-1], seq[1:]):
            trans[a][b] += 1
        row = trans.get(last, Counter())
        s = sum(row.values())
        total = len(seq)
        uni = Counter(seq)
        scores = {}
        for v in vals:
            p_decay = df.get(v, 0.0) / tot
            if s > 0:
                p_markov = 0.6 * (row.get(v, 0) / s) + 0.4 * (uni.get(v, 0) / total)
            else:
                p_markov = uni.get(v, 0) / total if total else 0.0
            scores[v] = w_prior * prior[v] + w_freq * p_decay + w_markov * p_markov
    # 均值回归信号: 近期window期某值出现过多(>先验期望)→降分; 过少→加分
    # 让预测随近期走势翻动(解决"期期一样"), 同时利用均值回归现象提升命中
    if revert_w > 0 and train and len(train) >= 5:
        revert_window = 10
        recent_vals = [extract(r) for r in train[-revert_window:] if extract(r) is not None]
        rn = len(recent_vals) or 1
        rc = Counter(recent_vals)
        for v in vals:
            expected = prior.get(v, 1.0 / len(vals)) * rn
            ratio = rc.get(v, 0) / expected if expected > 0 else 1.0
            revert_factor = max(0.3, min(2.0, 1.0 / max(ratio, 0.1)))
            scores[v] = scores[v] * (1 - revert_w + revert_w * revert_factor)
    mx = max(scores.values()) if scores else 0.0
    if mx <= 0:
        return {}
    return {v: s / mx for v, s in scores.items()}


def _dim_predict_value(records, extract, prior, cfg):
    """单维预测值: 衰减贝叶斯+马尔可夫混合 argmax; prior 为空回退全历史众数。"""
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


# 维度算法配置: name -> (top_k, algo)
#   algo="stacking": GBDT 分类器; algo="bayes": 衰减贝叶斯+马尔可夫混合
# 头数 top-2(命中率40%), 尾数 top-3(命中率40%, lift+10%); 其余 top-1。
# 选中频率惩罚: 尾数/头数等集合维度, 近PEN_LOOKBACK期已频繁选入top-k的
# 数字乘 penalty^cnt 降分, 避免某些数字"永远锁定"(如尾1/6恒定出现)。
# 仅影响 top-k 选择, 不改底层贝叶斯打分; lb=5 pen=0.80 实测打破锁定且命中46.7%。
PEN_LOOKBACK = 5
PEN_FACTOR = 0.80
PEN_DIMS = {"尾数", "头数"}  # 只对集合维度生效(top-k>1)

_DIM_CONFIG = {
    "波色": (2, "bayes"),     # 贝叶斯+均值回归(灵活): 近30期u3, 命中37%
    "生肖": (2, "bayes"),     # 贝叶斯(先验主导)
    "尾数": (4, "bayes"),     # top-4: 55%命中, lift+15%
    "大小": (1, "bayes"),     # 贝叶斯+均值回归: 命中47%
    "奇偶": (1, "bayes"),     # 贝叶斯+均值回归: 近30期u2, 命中77%
    "头数": (3, "bayes"),     # top-3: 贝叶斯+均值回归
}


def predict_dimensions(records, llm_result=None, backtest_n: int = 30) -> list:
    """六维独立预测 + 留一回测。

    波色/大小/奇偶: Stacking(GBDT) top-1。
    头数: Stacking(GBDT) top-2(集合命中率 40%)。
    生肖: 衰减贝叶斯 top-1。
    尾数: 衰减贝叶斯 top-3(集合命中率 40%, lift +10%)。
    基线: 预测集合的精确组合先验之和(与预测同口径, 无前视)。
    """
    import math
    zodiac_map = build_zodiac_map(records) if records else {}
    joint_n = _joint_predict_number(records, zodiac_map)
    dim_map = {name: ex for name, ex in DIMENSIONS}
    dim_data = {}  # name -> (pred_list, acc, baseline, se, stab, strategy)

    # ---- Stacking 维度 ----
    stacking_names = [n for n, (k, a) in _DIM_CONFIG.items() if a == "stacking"]
    if stacking_names:
        try:
            from dim_ensemble import (
                predict_stacking_value, backtest_stacking,
                predict_stacking_topk, backtest_stacking_topk,
                clear_cache as _dclear, DIM_PRIOR_FN)
            _dclear()
            for name in stacking_names:
                k, _ = _DIM_CONFIG[name]
                extract = dim_map[name]
                prior_fn = DIM_PRIOR_FN[name]
                if k == 1:
                    pred_list = [predict_stacking_value(records, extract, prior_fn, zodiac_map)]
                    hits = backtest_stacking(records, extract, prior_fn, backtest_n)
                else:
                    pred_list = predict_stacking_topk(records, extract, prior_fn, zodiac_map, k)
                    hits = backtest_stacking_topk(records, extract, prior_fn, backtest_n, k)
                n_h = len(hits)
                acc = sum(hits) / n_h if n_h else 0.0
                prior_now = comb_prior(name, zodiac_map)
                baseline = sum(prior_now.get(v, 0) for v in pred_list) if prior_now else k / 12
                se = math.sqrt(acc * (1 - acc) / n_h) if n_h else 0.0
                hits_dummy = [1] * int(acc * n_h) + [0] * (n_h - int(acc * n_h))
                stab = rolling_stability(hits_dummy) if n_h else 0.0
                dim_data[name] = (pred_list, acc, baseline, se, stab,
                                  f"Stacking非线性集成(GBDT分类器); top-{k}")
        except Exception:
            pass  # 降级到贝叶斯

    # ---- 贝叶斯维度(含 top-k) ----
    n_clamped = max(1, min(backtest_n, len(records) - 10))
    bayes_names = [n for n, (k, a) in _DIM_CONFIG.items()
                   if a == "bayes" and n not in dim_data]
    for name in bayes_names:
        k, _ = _DIM_CONFIG[name]
        extract = dim_map[name]
        use_penalty = name in PEN_DIMS and k > 1
        hits, bases = [], []
        sel_history = []  # 每期的top-k列表, 供选中频率惩罚使用
        for i in range(n_clamped):
            split = len(records) - n_clamped + i
            train = records[:split]
            actual = records[split]
            train_zmap = build_zodiac_map(train)
            prior = comb_prior(name, train_zmap)
            if not prior:
                continue
            scores = _posterior_scores(train, extract, prior, DIM_CONFIG.get(name))
            if use_penalty:
                # 选中频率惩罚: 近PEN_LOOKBACK期已选入top-k的数字降分
                recent_sel = Counter()
                for prev in sel_history[-PEN_LOOKBACK:]:
                    for v in prev:
                        recent_sel[v] += 1
                penalized = {v: s * (PEN_FACTOR ** recent_sel.get(v, 0))
                             for v, s in scores.items()}
                ranked = sorted(prior.keys(), key=lambda v: penalized.get(v, 0), reverse=True)
            else:
                ranked = sorted(prior.keys(), key=lambda v: scores.get(v, 0), reverse=True)
            topk = ranked[:k]
            sel_history.append(topk)
            hits.append(1 if extract(actual) in set(topk) else 0)
            bases.append(sum(prior.get(v, 0) for v in topk))
        acc = sum(hits) / len(hits) if hits else 0.0
        baseline = sum(bases) / len(bases) if bases else 0.0
        se = math.sqrt(acc * (1 - acc) / len(hits)) if hits else 0.0
        stab = rolling_stability(hits)
        prior_now = comb_prior(name, zodiac_map)
        scores_now = _posterior_scores(records, extract, prior_now, DIM_CONFIG.get(name))
        if use_penalty:
            # 当前预测也用相同惩罚: 用回测最后PEN_LOOKBACK期的历史
            recent_sel = Counter()
            for prev in sel_history[-PEN_LOOKBACK:]:
                for v in prev:
                    recent_sel[v] += 1
            penalized = {v: s * (PEN_FACTOR ** recent_sel.get(v, 0))
                         for v, s in scores_now.items()}
            ranked_now = sorted(prior_now.keys(), key=lambda v: penalized.get(v, 0), reverse=True)
        else:
            ranked_now = sorted(prior_now.keys(), key=lambda v: scores_now.get(v, 0), reverse=True)
        pred_list = ranked_now[:k]
        strat = f"衰减贝叶斯+马尔可夫混合+均值回归; top-{k}"
        if use_penalty:
            strat += f"+选中频率惩罚(近{PEN_LOOKBACK}期×{PEN_FACTOR})"
        dim_data[name] = (pred_list, acc, baseline, se, stab, strat)

    # ---- 组装结果 ----
    results = []
    for name, extract in DIMENSIONS:
        k, _ = _DIM_CONFIG[name]
        mode = _freq_predict(records, extract)
        if name in dim_data:
            pred_list, acc, baseline, se, stab, strategy = dim_data[name]
        else:
            # 降级兜底: 全历史众数
            pred_list = [_freq_predict(records, extract)]
            acc = baseline = se = stab = 0.0
            strategy = "降级(全历史众数)"
        value_str = "、".join(str(v) for v in pred_list if v is not None)
        results.append(DimensionPrediction(
            name=name,
            value=value_str,
            mode_value=str(mode) if mode is not None else "",
            joint_number=joint_n,
            strategy=strategy,
            accuracy=acc, baseline=baseline, lift=acc - baseline,
            std_error=se, stability=stab,
            value_set=[v for v in pred_list if v is not None],
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
                        k: int = ZODIAC_POOL_SIZE, use_stacking: bool = True) -> ZodiacPool:
    """生成单组 k 个最可能开出特码的生肖, 并回测。

    use_stacking=True(默认): 采用 Stacking(GBDT 元学习器)非线性集成,
    用 6 路 base 信号概率作为特征, GBDT 学习信号→命中的非线性映射。
    876 期实测优于线性加权; 数据不足时自动降级号码数先验。
    use_stacking=False: 回退号码数先验(快, 无 sklearn 依赖)。
    """
    import math
    baseline = k / 12

    if use_stacking:
        try:
            from zodiac_ensemble import predict_stacking_zodiacs, backtest_stacking
            zodiacs = predict_stacking_zodiacs(records, k)
            all_hits = backtest_stacking(records, backtest_n)
            hits = all_hits.get(k, [])
            n_clamped = len(hits)
            accuracy = sum(hits) / n_clamped if n_clamped else 0.0
            std_error = math.sqrt(accuracy * (1 - accuracy) / n_clamped) if n_clamped else 0.0
            stab = rolling_stability(hits)
            return ZodiacPool(
                zodiacs=zodiacs,
                strategy=f"Stacking非线性集成(GBDT); 6信号概率→P(命中); {n_clamped}期walk-forward回测",
                hit_rate=accuracy, baseline=baseline, lift=accuracy - baseline,
                std_error=std_error, stability=stab,
            )
        except Exception:
            pass  # 降级到号码数先验

    zodiac_map = build_zodiac_map(records) if records else {}
    zodiacs = _build_zodiac_pool(records, k, zodiac_map=zodiac_map, llm_result=llm_result)
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
        strategy="号码数组合先验(胖5/瘦4); Stacking降级",
        hit_rate=accuracy, baseline=baseline, lift=accuracy - baseline,
        std_error=std_error, stability=stab,
    )


def predict_zodiac_quad(records, llm_result=None, backtest_n: int = 30, use_stacking: bool = True) -> ZodiacPool:
    """生成单组 4 个最可能开出特码的生肖, 并回测。"""
    return predict_zodiac_pool(records, llm_result=llm_result, backtest_n=backtest_n, k=ZODIAC_QUAD_SIZE, use_stacking=use_stacking)


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


# 生肖集合打分信号注册表: (名称, 函数)。顺序即累加顺序; 增删/重排在此处操作。
#
# 为何只保留 numcount + llm(其余经验信号全部移除):
#   walk-forward 876 期实测, 各信号单独选六肖的命中(基线 6/12=50%):
#     freq 48.6% / markov 48.5% / cross_dim 48.4% / recent 47.6% / bayes 48.5%
#     numcount 51.6%  ← 唯一 > 基线
#   彩票特码生肖开奖近似独立均匀, freq/markov/recent/bayes/cross_dim 这些经验信号
#   追逐的是随机噪声, 叠加后系统性地把唯一真信号(号码数组合先验)淹没, 导致三/四/六肖
#   集合命中全部低于基线(旧 -0.5%~-1.3%)。移除它们后实测大幅回升:
#     三肖 24.5%→28.3% / 四肖 32.4%→36.2% / 六肖 48.7%→53.4% (全部 > 基线)。
#   原理: 号码数(胖生肖5/瘦生肖4)是特码生肖唯一的结构性先验概率差异, 年度内恒定且
#   无噪声; 经验频率则是大数律未收敛的随机涨落, 样本外无预测力。保留这些 _signal_xxx
#   函数体仅为可追溯/可回退, 不再注册生效。
_ZODIAC_SIX_SIGNALS = [
    ("freq", _signal_freq),
    ("markov", _signal_markov),
    ("cross_dim", _signal_cross_dim),
    ("recent", _signal_recent),
    ("bayes", _signal_bayes),
    ("llm", _signal_llm),
    ("numcount", _signal_numcount),
]


# 反持续性(生肖层): 上期开出生肖(seq[-1])累加得分乘此系数, 降低其进入本期三/四/六肖。
# 1.0=关闭, 0.0=等同禁绝; 默认 0.5 与号码层 _LAST_SPECIAL_DISCOUNT 同值。
# 号码大集合 _zodiac_bonus 走同一打分, 上期生肖跌出 top6 -> 其号码不再拿 bonus;
# 号码层 _wide_score 另对"上期生肖号码群"同口径折扣, 双层抑制"上期生肖重现"。
_LAST_ZODIAC_DISCOUNT = 0.5


def _zodiac_six_scores(records, llm_result=None, zodiac_map=None, llm_weight=0.0,
                       discount_last=_LAST_ZODIAC_DISCOUNT):
    """融合多种算法为每个生肖打分: 频率、马尔可夫(backoff)、跨维度融合、
    近期热度、贝叶斯后验、LLM信号、号码数加权。

    各信号实现见 _signal_xxx; 通过 _ZODIAC_SIX_SIGNALS 注册表按序累加,
    仅累加到出现在历史序列(all_zodiacs)中的生肖, 其余丢弃。
    records 透传给 ctx 供跨维度信号(_signal_cross_dim)复用。

    llm_weight: LLM 信号开关(回测恒 0, 预测可开), 详见 _signal_llm。
    discount_last: 反持续--上期开出生肖(seq[-1])累加得分乘此系数。回测每折
        train[-1] 即该折"上期", 无前视; _zodiac_bonus 同口径调用, 连带抑制
        上期生肖号码进入号码大集合。1.0=关闭(对照用)。
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
    # 反持续性: 上期开出生肖得分折扣(降低其在三/四/六肖及号码集合的出现)
    if discount_last != 1.0 and seq[-1] in scores:
        scores[seq[-1]] *= discount_last
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
                       llm_weight: float = 0.0, use_stacking: bool = True) -> ZodiacPool:
    """生成单组 6 个最可能开出特码的生肖, 并回测。

    use_stacking=True(默认): 采用 Stacking(GBDT 元学习器)非线性集成,
    用 6 路 base 信号概率作为特征, GBDT 学习信号→命中的非线性映射,
    输出每个生肖 P(命中), 取 top-6。876 期 walk-forward 实测六肖 50.9%(基线50%)。
    use_stacking=False: 回退号码数先验(胖5/瘦4)。

    llm_weight: LLM 信号开关。LLM 无法回测, 回测恒用 0; 预测默认亦为 0。
    无论是否参与选号, 只要传入 llm_result 即在 strategy 末尾附"大模型参考"展示。
    """
    import math
    baseline = ZODIAC_SIX_SIZE / 12
    zodiac_map = build_zodiac_map(records) if records else {}

    if use_stacking:
        try:
            from zodiac_ensemble import predict_stacking_zodiacs, backtest_stacking
            zodiacs = predict_stacking_zodiacs(records, ZODIAC_SIX_SIZE)
            all_hits = backtest_stacking(records, backtest_n)
            hits = all_hits.get(6, [])
            n_clamped = len(hits)
            hit_rate = sum(hits) / n_clamped if n_clamped else 0.0
            std_error = math.sqrt(hit_rate * (1 - hit_rate) / n_clamped) if n_clamped else 0.0
            stability = rolling_stability(hits)
            strategy = f"Stacking非线性集成(GBDT元学习器); 6信号概率→P(命中)→top6; {n_clamped}期walk-forward"
            if llm_weight:
                strategy += "+LLM信号"
            llm_ref = _llm_reference_zodiacs(llm_result, zodiac_map)
            if llm_ref:
                strategy += f"（大模型参考: {llm_ref}）"
            return ZodiacPool(
                zodiacs=zodiacs, strategy=strategy, hit_rate=hit_rate,
                baseline=baseline, lift=hit_rate - baseline,
                std_error=std_error, stability=stability,
            )
        except Exception:
            pass  # 降级号码数先验

    zodiacs = _build_zodiac_six(records, llm_result, zodiac_map, llm_weight)
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
    strategy = "号码数组合先验(胖5/瘦4); Stacking降级"
    llm_ref = _llm_reference_zodiacs(llm_result, zodiac_map)
    if llm_ref:
        strategy += f"（大模型参考: {llm_ref}）"
    return ZodiacPool(
        zodiacs=zodiacs, strategy=strategy, hit_rate=hit_rate,
        baseline=baseline, lift=hit_rate - baseline,
        std_error=std_error, stability=stability,
    )
