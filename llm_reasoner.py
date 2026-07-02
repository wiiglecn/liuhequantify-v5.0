#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""大模型反推理层 — OpenAI 兼容接口"""
import os
import sys
import json
import ssl
import re
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
    # 定位三个区段标记(下期预测 / 推理), 用位置切割避免正则回溯歧义
    p_match = re.search(r"【?下期预测】?", text)
    p_start = p_match.end() if p_match else len(text)

    # 推理标记: "【推理】" 或行首 "推理" (排除"反推理")
    r_match = re.search(r"(?:【推理】|^推理)", text, re.M)

    models_text = text[:p_start]
    if r_match and r_match.start() >= p_start:
        pred_text = text[p_start:r_match.start()]
        reasoning = text[r_match.end():].strip()
    else:
        pred_text = text[p_start:]
        reasoning = ""

    # 模型区: 提取编号行 "N. 名称"
    models = []
    for line in models_text.splitlines():
        line = line.strip()
        m = re.match(r"^\d+[.、)]\s*(.+)", line)
        if m:
            name = re.split(r"[：:]", m.group(1), maxsplit=1)[0].strip()
            if name:
                models.append(name)

    # 预测区: 提取 1-49 范围号码
    predicted = [int(x) for x in re.findall(r"\d{1,2}", pred_text)
                 if 1 <= int(x) <= 49]

    if not reasoning:
        reasoning = text.strip()

    return LLMResult(inferred_models=models, predicted_set=predicted, reasoning=reasoning)


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
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json; charset=utf-8",
                 "Authorization": "Bearer " + config["api_key"]},
    )
    with urllib.request.urlopen(req, context=_ssl_context(), timeout=60) as resp:
        result = json.loads(resp.read().decode("utf-8", "ignore"))
    text = result["choices"][0]["message"]["content"]
    return parse_llm_response(text)
