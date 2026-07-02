#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
六合彩网站预测内容准确率分析工具
自动抓取网站预测数据，对比历史开奖结果，计算各节点准确率并排名
"""

import urllib.request
import ssl
import re
import json
import os
import sys
from datetime import datetime
from collections import defaultdict

# 修复 Windows 控制台编码
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        os.system("chcp 65001 >nul 2>&1")


# ======================== 配置 ========================
WEBSITE_URL = "https://jgf-c4.jgfff.xyz:9385/?_360safeparam=36802359"
YJJY_URL = "https://jgf-c4.jgfff.xyz:9385/yjjy"
DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "history_data.json")

# 生肖与号码对应关系 (六合彩 1-49)
ZODIAC_MAP = {
    "鼠": [1, 13, 25, 37, 49],
    "牛": [2, 14, 26, 38],
    "虎": [3, 15, 27, 39],
    "兔": [4, 16, 28, 40],
    "龙": [5, 17, 29, 41],
    "蛇": [6, 18, 30, 42],
    "马": [7, 19, 31, 43],
    "羊": [8, 20, 32, 44],
    "猴": [9, 21, 33, 45],
    "鸡": [10, 22, 34, 46],
    "狗": [11, 23, 35, 47],
    "猪": [12, 24, 36, 48],
}

# 生肖排序 (用于年份计算)
ZODIAC_ORDER = ["鼠", "牛", "虎", "兔", "龙", "蛇", "马", "羊", "猴", "鸡", "狗", "猪"]


def get_ssl_context():
    """创建SSL上下文（跳过证书验证）"""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def fetch_url(url, encoding="utf-8"):
    """抓取网页内容"""
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    })
    try:
        resp = urllib.request.urlopen(req, context=get_ssl_context(), timeout=15)
        raw = resp.read()
        # 尝试多种编码
        for enc in [encoding, "utf-8", "gbk", "gb18030"]:
            try:
                return raw.decode(enc)
            except (UnicodeDecodeError, LookupError):
                continue
        return raw.decode("utf-8", errors="replace")
    except Exception as e:
        print(f"[错误] 抓取 {url} 失败: {e}")
        return None


def number_to_zodiac(num):
    """根据号码获取生肖"""
    if num < 1 or num > 49:
        return None
    # 2025年是蛇年，号码1对应蛇
    # 生肖循环: 蛇(1) 马(2) 羊(3) 猴(4) 鸡(5) 狗(6) 猪(7) 鼠(8) 牛(9) 虎(10) 兔(11) 龙(12)
    # 然后 13 又是蛇, 14 马...
    year_offset = (num - 1) % 12
    zodiac_index = (4 + year_offset) % 12  # 2025年蛇年，蛇在index 4
    return ZODIAC_ORDER[zodiac_index]


def zodiac_to_numbers(zodiac):
    """获取生肖对应的号码列表"""
    return ZODIAC_MAP.get(zodiac, [])


# ======================== 数据解析 ========================

def parse_jinguangfo(html):
    """
    解析金光佛预测数据
    返回: [{period, source, zodiac_5, zodiac_3, zodiac_2, code_5, code_4, code_2, highlight_zodiac, highlight_code}]
    """
    records = []
    # 匹配每个金光佛五肖⑤码区块
    # HTML结构: <b>140期金光佛870567《五肖⑤码》</b>
    pattern = re.compile(
        r'<b>(\d+)期金光佛\d+《五肖⑤码》</b>.*?</tr>(.*?)(?=<tr>\s*<td[^>]*bgcolor=[\'"]#005aff[\'"]|$)',
        re.DOTALL
    )

    for m in pattern.finditer(html):
        period = m.group(1)
        content = m.group(2)

        record = {
            "period": period,
            "source": "金光佛870567",
            "type": "五肖⑤码",
            "zodiac_5": [],
            "zodiac_3": [],
            "zodiac_2": [],
            "code_5": [],
            "code_4": [],
            "code_2": [],
            "highlight_zodiac": [],
            "highlight_code": [],
        }

        # 提取五肖 - 从 td 内容中提取
        z5_match = re.search(r'五肖:<.*?font[^>]*>(.*?)</font>', content, re.DOTALL)
        if z5_match:
            z5_html = z5_match.group(1)
            z5_text = re.sub(r'<[^>]+>', '', z5_html).strip()
            # 提取高亮生肖
            z5_highlights = re.findall(r'<span[^>]*background-color:\s*#FFFF00[^>]*>([^<]+)</span>', z5_html)
            record["highlight_zodiac"] = [z.strip() for z in z5_highlights]
            # 提取所有生肖
            for z in ZODIAC_ORDER:
                if z in z5_text:
                    record["zodiac_5"].append(z)

        # 提取三肖
        z3_match = re.search(r'三肖:.*?font[^>]*>(.*?)</font>', content, re.DOTALL)
        if z3_match:
            z3_text = re.sub(r'<[^>]+>', '', z3_match.group(1)).strip()
            for z in ZODIAC_ORDER:
                if z in z3_text:
                    record["zodiac_3"].append(z)

        # 提取二肖
        z2_match = re.search(r'二肖:.*?font[^>]*>(.*?)</font>', content, re.DOTALL)
        if z2_match:
            z2_text = re.sub(r'<[^>]+>', '', z2_match.group(1)).strip()
            for z in ZODIAC_ORDER:
                if z in z2_text:
                    record["zodiac_2"].append(z)

        # 提取⑤码
        c5_match = re.search(r'⑤码:.*?font[^>]*>(.*?)</font>', content, re.DOTALL)
        if c5_match:
            c5_html = c5_match.group(1)
            c5_text = re.sub(r'<[^>]+>', '', c5_html).strip()
            nums = re.findall(r'\d+', c5_text)
            record["code_5"] = [int(n) for n in nums if 1 <= int(n) <= 49]
            # 提取高亮号码
            c5_highlights = re.findall(r'<span[^>]*background-color:\s*#FFFF00[^>]*>(\d+)</span>', c5_html)
            record["highlight_code"] = [int(n) for n in c5_highlights]

        # 提取④码
        c4_match = re.search(r'④码:.*?font[^>]*>(.*?)</font>', content, re.DOTALL)
        if c4_match:
            c4_text = re.sub(r'<[^>]+>', '', c4_match.group(1)).strip()
            nums = re.findall(r'\d+', c4_text)
            record["code_4"] = [int(n) for n in nums if 1 <= int(n) <= 49]

        # 提取②码
        c2_match = re.search(r'②码:.*?font[^>]*>(.*?)</font>', content, re.DOTALL)
        if c2_match:
            c2_text = re.sub(r'<[^>]+>', '', c2_match.group(1)).strip()
            nums = re.findall(r'\d+', c2_text)
            record["code_2"] = [int(n) for n in nums if 1 <= int(n) <= 49]

        if record["zodiac_5"] or record["code_5"]:
            records.append(record)

    return records


def parse_neimu_sanma(html):
    """
    解析广东内幕三码数据
    返回: [{period, source, numbers, result_zodiac, result_number, accurate}]
    """
    records = []
    # 期号可能在单独的 <font> 标签中，需要处理 HTML 标签分隔的情况
    # 匹配: <font...>144</font><font...>期;内幕三码<font...>【41.42.43】</font></font>开：鼠43准
    pattern = re.compile(
        r'<font[^>]*>(\d+)</font>\s*<font[^>]*>期;内幕三码<font[^>]*>【(.*?)】</font></font>开[：:]([^<\s]*?)(\d+)(准?)'
    )

    for m in pattern.finditer(html):
        period = m.group(1)
        nums_html = m.group(2)
        result_zodiac = m.group(3)
        result_number = int(m.group(4))
        accurate = m.group(5) == "准"

        # 从HTML中提取纯数字
        nums_text = re.sub(r'<[^>]+>', '', nums_html)
        numbers = [int(n) for n in re.findall(r'\d+', nums_text) if 1 <= int(n) <= 49]

        records.append({
            "period": period,
            "source": "广东内幕三码",
            "type": "内幕三码",
            "numbers": numbers,
            "result_zodiac": result_zodiac,
            "result_number": result_number,
            "accurate": accurate,
        })

    return records


def parse_pingte_yixiao(html):
    """
    解析金光佛平特一肖数据
    返回: [{period, source, zodiac, result_zodiac, result_number, accurate}]
    """
    records = []
    # 匹配模式: 141期-平特一肖【蛇蛇蛇】开：蛇02
    pattern = re.compile(
        r'(\d+)期-平特一肖<font[^>]*>【<span[^>]*>([^<]+)</span>】</font>开[：:]([^<\s]+?)(\d+)'
    )

    for m in pattern.finditer(html):
        period = m.group(1)
        zodiac_pred = re.sub(r'(.)(\1+)', r'\1', m.group(2))  # 去重: 蛇蛇蛇 -> 蛇
        result_zodiac = m.group(3)
        result_number = int(m.group(4))

        accurate = zodiac_pred == result_zodiac

        records.append({
            "period": period,
            "source": "金光佛平特一肖",
            "type": "平特一肖",
            "zodiac": zodiac_pred,
            "result_zodiac": result_zodiac,
            "result_number": result_number,
            "accurate": accurate,
        })

    return records


def parse_yixiao_erma(html):
    """
    解析一肖②码数据
    返回: [{period, source, zodiac, numbers, result_zodiac, result_number, accurate}]
    """
    records = []
    # 匹配模式: <font...>144期;一肖②码<font color="#FF0000">【<span...>鼠</span>+18.17】</font></font>开:鼠43准
    pattern = re.compile(
        r'<font[^>]*>(\d+)期;一肖②码<font[^>]*>【<span[^>]*>([^<]+)</span>\+([\d.]+)】</font></font>开[：:]([^<\s]*?)(\d+)(准?)'
    )

    for m in pattern.finditer(html):
        period = m.group(1)
        zodiac = m.group(2)
        nums_str = m.group(3)
        result_zodiac = m.group(4)
        result_number = int(m.group(5))
        accurate = m.group(6) == "准"

        numbers = [int(n) for n in nums_str.split(".") if n.isdigit()]

        # 检查生肖是否准确
        zodiac_accurate = zodiac == result_zodiac
        # 检查号码是否准确
        number_accurate = result_number in numbers

        records.append({
            "period": period,
            "source": "一肖②码",
            "type": "一肖②码",
            "zodiac": zodiac,
            "numbers": numbers,
            "result_zodiac": result_zodiac,
            "result_number": result_number,
            "zodiac_accurate": zodiac_accurate,
            "number_accurate": number_accurate,
            "accurate": accurate,
        })

    return records


def parse_guanggao_tuijian(html):
    """
    解析广告推荐位数据（提取各来源名称）
    返回: [{source_name, period, description}]
    """
    records = []
    pattern = re.compile(
        r'第(\d+)期:(.*?)</font>'
    )
    for m in pattern.finditer(html):
        period = m.group(1)
        desc = m.group(2).strip()
        # 提取来源名称
        source_match = re.search(r'【(.+?)】', desc)
        source_name = source_match.group(1) if source_match else desc[:20]
        records.append({
            "period": period,
            "source_name": source_name,
            "description": desc,
        })
    return records


# ======================== 准确率计算 ========================

def calculate_accuracy(all_records):
    """
    计算各来源/类型的准确率
    返回: 按准确率排序的列表
    """
    source_stats = defaultdict(lambda: {
        "total": 0,
        "accurate": 0,
        "zodiac_correct": 0,
        "number_correct": 0,
        "records": [],
    })

    for rec in all_records:
        source = rec.get("source", "未知")
        key = f"{source} ({rec.get('type', '')})"
        stats = source_stats[key]
        stats["total"] += 1
        stats["records"].append(rec)

        if rec.get("accurate"):
            stats["accurate"] += 1

        # 生肖准确率
        if rec.get("zodiac_accurate") or rec.get("result_zodiac"):
            pred_zodiac = rec.get("zodiac") or rec.get("result_zodiac")
            result_zodiac = rec.get("result_zodiac")
            if pred_zodiac and result_zodiac and pred_zodiac == result_zodiac:
                stats["zodiac_correct"] += 1

        # 号码准确率
        if rec.get("number_accurate"):
            stats["number_correct"] += 1
        elif rec.get("numbers") and rec.get("result_number"):
            if rec["result_number"] in rec["numbers"]:
                stats["number_correct"] += 1
        elif rec.get("code_5") and rec.get("result_number"):
            if rec["result_number"] in rec["code_5"]:
                stats["number_correct"] += 1

    # 计算准确率并排序
    results = []
    for key, stats in source_stats.items():
        if stats["total"] == 0:
            continue
        accuracy = stats["accurate"] / stats["total"] * 100
        zodiac_acc = stats["zodiac_correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        number_acc = stats["number_correct"] / stats["total"] * 100 if stats["total"] > 0 else 0

        results.append({
            "source": key,
            "total": stats["total"],
            "accurate": stats["accurate"],
            "accuracy": accuracy,
            "zodiac_accuracy": zodiac_acc,
            "number_accuracy": number_acc,
            "records": stats["records"],
        })

    results.sort(key=lambda x: x["accuracy"], reverse=True)
    return results


def analyze_zodiac_hit_rate(all_records):
    """分析各生肖的命中率"""
    zodiac_stats = defaultdict(lambda: {"total": 0, "hit": 0})

    for rec in all_records:
        pred_zodiacs = rec.get("zodiac_5", []) or ([rec["zodiac"]] if rec.get("zodiac") else [])
        result_zodiac = rec.get("result_zodiac")

        if not result_zodiac or not pred_zodiacs:
            continue

        zodiac_stats[result_zodiac]["total"] += 1
        if result_zodiac in pred_zodiacs:
            zodiac_stats[result_zodiac]["hit"] += 1

    results = []
    for zodiac, stats in zodiac_stats.items():
        if stats["total"] > 0:
            rate = stats["hit"] / stats["total"] * 100
            results.append({
                "zodiac": zodiac,
                "total": stats["total"],
                "hit": stats["hit"],
                "rate": rate,
            })

    results.sort(key=lambda x: x["rate"], reverse=True)
    return results


def analyze_number_frequency(all_records):
    """分析号码出现频率"""
    number_stats = defaultdict(int)
    for rec in all_records:
        nums = rec.get("numbers", []) or rec.get("code_5", [])
        for n in nums:
            number_stats[n] += 1

    results = [{"number": n, "count": c} for n, c in number_stats.items()]
    results.sort(key=lambda x: x["count"], reverse=True)
    return results


# ======================== 交叉匹配开奖结果 ========================

def match_cross_source_results(all_records):
    """
    从有开奖结果的来源中提取结果，补充到没有结果的记录中
    例如: 五肖⑤码没有开奖结果，但从平特一肖、内幕三码等获取
    """
    # 1. 建立期号 -> 开奖结果的查找表
    result_lookup = {}
    for rec in all_records:
        period = rec.get("period")
        result_zodiac = rec.get("result_zodiac")
        result_number = rec.get("result_number")
        if period and result_zodiac and result_number and result_zodiac != "?" and result_number != 0:
            if period not in result_lookup:
                result_lookup[period] = {
                    "result_zodiac": result_zodiac,
                    "result_number": result_number,
                }

    # 2. 为没有开奖结果的记录补充结果
    matched_count = 0
    for rec in all_records:
        period = rec.get("period")
        # 如果记录没有开奖结果，尝试从查找表获取
        if period and (not rec.get("result_zodiac") or rec.get("result_zodiac") == "?" or not rec.get("result_number") or rec.get("result_number") == 0):
            if period in result_lookup:
                rec["result_zodiac"] = result_lookup[period]["result_zodiac"]
                rec["result_number"] = result_lookup[period]["result_number"]
                matched_count += 1

                # 重新计算准确率
                # 对于五肖⑤码：检查预测的生肖是否包含开奖结果的生肖
                if rec.get("zodiac_5"):
                    result_z = result_lookup[period]["result_zodiac"]
                    rec["accurate"] = result_z in rec["zodiac_5"]
                    # 检查高亮生肖是否命中
                    if rec.get("highlight_zodiac"):
                        rec["highlight_accurate"] = result_z in rec["highlight_zodiac"]
                # 对于内幕三码：检查号码是否命中
                if rec.get("numbers"):
                    result_n = result_lookup[period]["result_number"]
                    rec["accurate"] = result_n in rec["numbers"]

    return matched_count


# ======================== 数据持久化 ========================

def save_data(records):
    """保存历史数据"""
    data = []
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

    # 合并数据（去重）
    existing_keys = set()
    for d in data:
        key = f"{d.get('period')}_{d.get('source')}_{d.get('type')}"
        existing_keys.add(key)

    for rec in records:
        key = f"{rec.get('period')}_{rec.get('source')}_{rec.get('type')}"
        if key not in existing_keys:
            data.append(rec)
            existing_keys.add(key)

    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return len(data)


def load_history():
    """加载历史数据"""
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


# ======================== 显示 ========================

def print_header(title):
    """打印标题"""
    width = 70
    print("\n" + "=" * width)
    print(f"{'':>20}{title}")
    print("=" * width)


def print_accuracy_ranking(results):
    """打印准确率排名"""
    print_header("📊 各预测来源准确率排名")

    if not results:
        print("  暂无数据")
        return

    print(f"\n  {'排名':<4} {'来源':<25} {'总期数':<6} {'准确':<6} {'准确率':<8} {'生肖准确率':<10} {'号码准确率':<10}")
    print("  " + "-" * 80)

    for i, r in enumerate(results, 1):
        acc_bar = "█" * int(r["accuracy"] / 5) + "░" * (20 - int(r["accuracy"] / 5))
        print(f"  {i:<4} {r['source']:<25} {r['total']:<6} {r['accurate']:<6} {r['accuracy']:<7.1f}% {r['zodiac_accuracy']:<9.1f}% {r['number_accuracy']:<9.1f}%")

    # 高准确率来源
    high_acc = [r for r in results if r["accuracy"] >= 70]
    if high_acc:
        print(f"\n  ✅ 高准确率来源 (≥70%):")
        for r in high_acc:
            print(f"     • {r['source']}: {r['accuracy']:.1f}% ({r['accurate']}/{r['total']})")


def print_zodiac_analysis(zodiac_results):
    """打印生肖分析"""
    print_header("🐉 各生肖命中率分析")

    if not zodiac_results:
        print("  暂无数据")
        return

    print(f"\n  {'生肖':<6} {'出现次数':<8} {'命中次数':<8} {'命中率':<8} {'柱状图'}")
    print("  " + "-" * 60)

    for r in zodiac_results:
        bar = "█" * int(r["rate"] / 5) + "░" * (20 - int(r["rate"] / 5))
        print(f"  {r['zodiac']:<6} {r['total']:<8} {r['hit']:<8} {r['rate']:<7.1f}% {bar}")


def print_number_frequency(freq_results):
    """打印号码频率"""
    print_header("🔢 热门号码排行 (Top 15)")

    if not freq_results:
        print("  暂无数据")
        return

    print(f"\n  {'排名':<4} {'号码':<6} {'出现次数':<8} {'频率条'}")
    print("  " + "-" * 50)

    max_count = freq_results[0]["count"] if freq_results else 1
    for i, r in enumerate(freq_results[:15], 1):
        bar_len = int(r["count"] / max_count * 20)
        bar = "█" * bar_len
        print(f"  {i:<4} {r['number']:<6} {r['count']:<8} {bar}")


def print_recent_predictions(records):
    """打印最近几期的预测详情"""
    print_header("📋 最近预测详情")

    # 按期号分组
    by_period = defaultdict(list)
    for rec in records:
        by_period[rec.get("period", "?")].append(rec)

    # 排序（降序）
    sorted_periods = sorted(by_period.keys(), key=lambda x: int(x) if x.isdigit() else 0, reverse=True)

    for period in sorted_periods[:5]:  # 最近5期
        recs = by_period[period]
        print(f"\n  ── 第 {period} 期 ──")
        for rec in recs:
            source = rec.get("source", "")
            zodiacs = rec.get("zodiac_5", []) or rec.get("zodiac", "")
            if isinstance(zodiacs, list):
                zodiacs = " ".join(zodiacs)
            numbers = rec.get("numbers", []) or rec.get("code_5", [])
            result = f"{rec.get('result_zodiac', '?')}{rec.get('result_number', '?')}"
            acc = "✓" if rec.get("accurate") else "✗"

            print(f"    [{source}] 预测: 生肖={zodiacs}, 号码={numbers} → 开奖: {result} {acc}")


def print_high_accuracy_nodes(results):
    """打印高准确率节点"""
    print_header("🎯 高准确率节点整理 (按准确率降序)")

    if not results:
        print("  暂无数据")
        return

    # 按准确率分组
    tiers = {
        "🔥 极高准确率 (≥90%)": [],
        "✅ 高准确率 (70-89%)": [],
        "📊 中等准确率 (50-69%)": [],
        "⚠️ 低准确率 (<50%)": [],
    }

    for r in results:
        acc = r["accuracy"]
        if acc >= 90:
            tiers["🔥 极高准确率 (≥90%)"].append(r)
        elif acc >= 70:
            tiers["✅ 高准确率 (70-89%)"].append(r)
        elif acc >= 50:
            tiers["📊 中等准确率 (50-69%)"].append(r)
        else:
            tiers["⚠️ 低准确率 (<50%)"].append(r)

    for tier_name, tier_records in tiers.items():
        if not tier_records:
            continue
        print(f"\n  {tier_name}")
        print("  " + "-" * 60)
        for r in tier_records:
            print(f"    • {r['source']}")
            print(f"      准确率: {r['accuracy']:.1f}% | 总期数: {r['total']} | 准确: {r['accurate']}")
            print(f"      生肖准确率: {r['zodiac_accuracy']:.1f}% | 号码准确率: {r['number_accuracy']:.1f}%")

            # 显示最近记录
            recent = r["records"][-3:]  # 最近3条
            if recent:
                print(f"      最近记录:")
                for rec in recent:
                    result = f"{rec.get('result_zodiac', '?')}{rec.get('result_number', '?')}"
                    acc_mark = "✓" if rec.get("accurate") else "✗"
                    print(f"        第{rec.get('period', '?')}期 → 开奖: {result} {acc_mark}")
            print()


# ======================== 主程序 ========================

def main():
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + "六合彩网站预测内容准确率分析工具".center(56) + "║")
    print("║" + f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(60) + "║")
    print("╚" + "═" * 68 + "╝")

    # 1. 抓取网站数据
    print("\n[1/5] 正在抓取网站数据...")
    html = fetch_url(YJJY_URL)
    if not html:
        print("无法获取网站数据，请检查网络连接")
        return

    print(f"  ✓ 成功获取页面 ({len(html)} 字节)")

    # 2. 解析各来源数据
    print("\n[2/5] 正在解析预测数据...")

    all_records = []

    # 金光佛数据
    jgf_records = parse_jinguangfo(html)
    print(f"  ✓ 金光佛870567: {len(jgf_records)} 条记录")
    all_records.extend(jgf_records)

    # 内幕三码
    neimu_records = parse_neimu_sanma(html)
    print(f"  ✓ 广东内幕三码: {len(neimu_records)} 条记录")
    all_records.extend(neimu_records)

    # 平特一肖
    pingte_records = parse_pingte_yixiao(html)
    print(f"  ✓ 金光佛平特一肖: {len(pingte_records)} 条记录")
    all_records.extend(pingte_records)

    # 一肖②码
    yixiao_records = parse_yixiao_erma(html)
    print(f"  ✓ 一肖②码: {len(yixiao_records)} 条记录")
    all_records.extend(yixiao_records)

    # 广告推荐位
    guanggao_records = parse_guanggao_tuijian(html)
    print(f"  ✓ 广告推荐位: {len(guanggao_records)} 条记录")

    if not all_records:
        print("\n  ⚠ 未解析到有效预测数据")
        return

    # 2.5 交叉匹配开奖结果（为五肖⑤码等补充开奖结果）
    matched = match_cross_source_results(all_records)
    if matched > 0:
        print(f"  ✓ 交叉匹配开奖结果: {matched} 条记录已补充")

    # 3. 保存数据
    print("\n[3/5] 正在保存历史数据...")
    total_saved = save_data(all_records)
    print(f"  ✓ 累计保存 {total_saved} 条记录")

    # 4. 计算准确率
    print("\n[4/5] 正在计算准确率...")
    accuracy_results = calculate_accuracy(all_records)
    zodiac_analysis = analyze_zodiac_hit_rate(all_records)
    number_freq = analyze_number_frequency(all_records)
    print(f"  ✓ 分析完成，共 {len(accuracy_results)} 个来源")

    # 5. 显示结果
    print("\n[5/5] 正在生成分析报告...\n")

    print_accuracy_ranking(accuracy_results)
    print_high_accuracy_nodes(accuracy_results)
    print_zodiac_analysis(zodiac_analysis)
    print_number_frequency(number_freq)
    print_recent_predictions(all_records)

    # 总结
    print_header("📝 分析总结")
    total = len(all_records)
    accurate = sum(1 for r in all_records if r.get("accurate"))
    overall_acc = accurate / total * 100 if total > 0 else 0

    print(f"\n  总预测记录数: {total}")
    print(f"  总准确次数: {accurate}")
    print(f"  整体准确率: {overall_acc:.1f}%")

    if accuracy_results:
        best = accuracy_results[0]
        print(f"\n  🏆 最准确来源: {best['source']}")
        print(f"     准确率: {best['accuracy']:.1f}% ({best['accurate']}/{best['total']})")

    high_acc = [r for r in accuracy_results if r["accuracy"] >= 70]
    if high_acc:
        print(f"\n  📌 建议关注的高准确率来源:")
        for r in high_acc[:5]:
            print(f"     • {r['source']}: {r['accuracy']:.1f}%")

    print(f"\n  数据已保存至: {DATA_FILE}")
    print("\n" + "=" * 70)
    print("  提示: 以上分析仅供参考，彩票开奖结果完全随机，请理性对待")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
