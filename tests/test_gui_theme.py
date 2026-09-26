# -*- coding: utf-8 -*-
"""GUI 主题令牌与绘制助手测试(纯函数部分不依赖显示器)。"""
import importlib

import pytest

mg = importlib.import_module("macau_gui")


def test_theme_tokens_are_hex():
    names = ["BG_MAIN", "BG_CARD", "BG_CARD2", "BG_SIDE", "BORDER", "BORDER_G",
             "SHADOW", "BALL_SHADOW", "PRIMARY", "PRIMARY_D", "PRIMARY_L",
             "PRIMARY_BG", "ACCENT", "INDIGO", "AMBER", "TEXT", "TEXT2",
             "MUTED", "OK", "WARN", "OK_BG", "OK_BORDER", "TRACK"]
    for name in names:
        v = getattr(mg, name)
        assert isinstance(v, str) and v.startswith("#") and len(v) == 7, name
        int(v[1:], 16)


def test_muted_is_soft_gray():
    # 新设计弱化文字为柔和灰(取代旧版纯黑)
    assert mg.MUTED == "#9aa1ad"


@pytest.mark.skipif(not mg._PIL_OK, reason="PIL 不可用")
def test_gradient_pil():
    img = mg._gradient_pil(60, 20, "#e11d48", "#fb7185", radius=6)
    assert img.size == (60, 20) and img.mode == "RGBA"
    px = img.load()
    assert px[0, 10][:3] == (225, 29, 72)          # 左端 == c1
    assert px[59, 10][:3] == (251, 113, 133)       # 右端 == c2
    assert px[0, 0][3] == 0                        # 圆角外透明
    assert px[2, 10][3] == 255                     # 中部不透明


@pytest.mark.skipif(not mg._PIL_OK, reason="PIL 不可用")
def test_shadow_pil():
    img = mg._shadow_pil(80, 40, radius=10)
    assert img.size == (80, 40) and img.mode == "RGBA"
    px = img.load()
    assert px[40, 36][3] > 0                       # 底部偏移处有影
    assert px[0, 0][3] < px[40, 36][3]             # 远角明显更淡
