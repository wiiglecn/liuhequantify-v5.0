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
    """加载多年记录。
    - refresh=True: 重新抓取所有年份 + 最新期。
    - refresh=False: 用缓存作底, 再抓取最新一期增量合并(保证拿到最新开奖)。
      最新一期抓取失败则回退缓存。
    """
    if refresh:
        cached = []
    else:
        cached = load_cache()

    lists = []
    if not cached:
        # 无缓存: 抓取所有年份
        for y in years:
            try:
                lists.append([parse_record(r) for r in fetch_year(y)])
            except Exception as e:
                print(f"  ! 抓取 {y} 年失败: {e}")
    else:
        lists.append(cached)

    # 始终抓取最新一期, 与缓存合并(增量更新)
    try:
        lists.append([parse_record(r) for r in fetch_latest()])
    except Exception as e:
        print(f"  ! 抓取最新期失败, 使用缓存: {e}")

    merged = merge_records(lists)
    if merged:
        save_cache(merged)
    return merged
