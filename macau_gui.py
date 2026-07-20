#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""澳门六合彩分析与预测 - 图形界面版(tkinter)
白底清爽主题 + 淡红主色调 + 3D 立体号码球 + 卡片流排版。"""
import os
import sys
import threading
import queue
from datetime import datetime

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        os.system("chcp 65001 >nul 2>&1")

import tkinter as tk
from tkinter import ttk, messagebox

# 可选图像库: 用于号码球高清渲染(超采样 + LANCZOS 抗锯齿); 缺失则降级矢量绘制
try:
    from PIL import Image, ImageDraw, ImageFilter, ImageTk
    import numpy as _np
    try:
        _LANCZOS = Image.Resampling.LANCZOS   # Pillow >= 10
    except AttributeError:                     # Pillow < 10
        _LANCZOS = Image.LANCZOS
    _PIL_OK = True
except Exception:
    _PIL_OK = False

from data_fetcher import load_history, load_cache, merge_records, save_cache, Record, compute_next_issue
from analysis import build_report, summarize_for_llm
from llm_reasoner import load_config, reason
from predictor import predict_all, predict_special_groups
from dimensions import (predict_dimensions, predict_zodiac_pool, predict_zodiac_quad, predict_zodiac_six,
                        wave_of_number,
                        build_zodiac_map)
from special_pool import predict_special_pools, predict_wide_pool
from prediction_store import (serialize_run, save_run, load_runs,
                              verify_run, delete_run, clear_runs)

# ======================== 主题配色 (白底清爽 + 淡红主色) ========================
BG_MAIN = "#ffffff"        # 页面背景(白)
BG_CARD = "#ffffff"        # 卡片背景(白)
BG_CARD2 = "#f0f2f5"       # 卡片次级/表头底
BORDER = "#f0d4d4"         # 卡片边框(淡红)
BORDER_G = "#e6e8eb"       # 通用边框(灰)
SHADOW = "#d8dce2"         # 卡片投影(灰)
PRIMARY = "#e57373"        # 淡红主色
PRIMARY_D = "#c05050"      # 深红
PRIMARY_L = "#f6b4b4"      # 浅红
ACCENT = "#e84545"         # 鲜红(高亮/推荐)
TEXT = "#2c2f36"           # 主文字(深灰)
TEXT2 = "#5a606b"          # 次文字
MUTED = "#000000"          # 弱化文字(按要求改为黑色)
OK = "#2ea043"             # 正向(lift>0)
WARN = "#d68910"           # 警示

FONT = ("Microsoft YaHei UI", 10)
FONT_SM = ("Microsoft YaHei UI", 9)
FONT_H = ("Microsoft YaHei UI", 12, "bold")
FONT_TITLE = ("Microsoft YaHei UI", 18, "bold")
FONT_NUM = ("Consolas", 12, "bold")
FONT_NUM_S = ("Consolas", 11, "bold")
FONT_ZOD = ("Microsoft YaHei UI", 12, "bold")

DISCLAIMER = (
    "科学提示: 彩票开奖本质为独立随机事件, 任何模型的真实命中率理论上均接近随机概率\n"
    "(特码 1/49≈2%, 单平码 6/49)。命中率基于历史回测, 不代表未来, 仅供研究, 不构成投注建议。\n"
    "算法模型：AI大模型反推理 + 数学模型 + 马尔可夫模型 + 贝叶斯后验"
)

# 抓取年份(供主分析与历史页刷新核对共用); 自动覆盖 2024 至当前年, 跨年无需手改
HISTORY_YEARS = list(range(2024, datetime.now().year + 1))


def _hex(s):
    return int(s, 16)


def _blend(c1, c2, t):
    r1, g1, b1 = _hex(c1[1:3]), _hex(c1[3:5]), _hex(c1[5:7])
    r2, g2, b2 = _hex(c2[1:3]), _hex(c2[3:5]), _hex(c2[5:7])
    r = int(r1 + (r2 - r1) * t)
    g = int(g1 + (g2 - g1) * t)
    b = int(b1 + (b2 - b1) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


def _hex_rgb(s):
    """#rrggbb -> (r, g, b) int 元组。"""
    return _hex(s[1:3]), _hex(s[3:5]), _hex(s[5:7])


# 号码球 PhotoImage 缓存: 同一(尺寸/波色/命中)的球复用同一张图, 避免重复渲染
_BALL_IMG_CACHE = {}


def _render_ball_pil(size, r, base, light, ring=None, scale=4):
    """PIL 超采样渲染一颗高清圆润立体球(RGBA, 透明底), 返回 PIL.Image。

    - 投影: 偏右下柔化半透明深色椭圆(地面落影)
    - 球体: 以真实圆心做圆形遮罩(边缘 1.5px 软边 → 抗锯齿),
      内部为偏左上径向渐变(边缘 base → 高光 light) + 镜面高光(趋向白)
    - 命中金环: 球体边缘内侧金色描边
    - 4 倍超采样后 LANCZOS 降回显示尺寸 → 边缘细腻、圆润、高清
    """
    size = int(round(size))
    S = size * scale
    R = float(r) * scale
    cx = cy = S / 2.0
    ys, xs = _np.mgrid[0:S, 0:S].astype(_np.float32)
    # 高光中心偏向左上(模拟左上方光源)
    hx, hy = cx - R * 0.30, cy - R * 0.30
    dl = _np.sqrt((xs - hx) ** 2 + (ys - hy) ** 2)
    # 主渐变: 边缘 base → 高光 light
    t = _np.clip(1.0 - dl / (R * 1.25), 0.0, 1.0) ** 1.4
    # 镜面高光: 高光中心附近趋向白色(小范围)
    ts = _np.clip(1.0 - dl / (R * 0.30), 0.0, 1.0) ** 2.2
    br, bg, bb = _hex_rgb(base)
    lr, lg, lb = _hex_rgb(light)
    rr = br + (lr - br) * t
    gg = bg + (lg - bg) * t
    bv = bb + (lb - bb) * t
    rr = rr + (255 - rr) * ts * 0.55
    gg = gg + (255 - gg) * ts * 0.55
    bv = bv + (255 - bv) * ts * 0.55
    rgb = _np.dstack([rr, gg, bv]).clip(0, 255).astype(_np.uint8)
    # 球体 alpha: 真实圆心距做圆形遮罩, 边缘 1.5px 软边
    dc = _np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    alpha = _np.clip((R - dc) / (1.5 * scale) + 0.5, 0.0, 1.0)
    rgba = _np.dstack([rgb, (alpha * 255).astype(_np.uint8)])
    sphere = Image.fromarray(rgba, "RGBA")
    # 投影(柔化)
    sh = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    ImageDraw.Draw(sh).ellipse(
        [cx - R + 2 * scale, cy - R + 3 * scale, cx + R + 2 * scale, cy + R + 3 * scale],
        fill=(*_hex_rgb(SHADOW), 90))
    sh = sh.filter(ImageFilter.GaussianBlur(1.1 * scale))
    out = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    out.alpha_composite(sh)
    out.alpha_composite(sphere)
    # 命中金环(画在球体边缘内侧)
    if ring:
        rw = max(2, int(round(2 * scale)))
        ImageDraw.Draw(out).ellipse(
            [cx - R + rw / 2, cy - R + rw / 2, cx + R - rw / 2, cy + R - rw / 2],
            outline=ring, width=rw)
    return out.resize((size, size), _LANCZOS)


def _get_ball_image(size, r, base, light, ring=None):
    """取(必要则渲染并缓存)一颗球的 PhotoImage。同一外观的球全局复用。"""
    key = (size, r, base, light, ring)
    img = _BALL_IMG_CACHE.get(key)
    if img is None:
        img = ImageTk.PhotoImage(_render_ball_pil(size, r, base, light, ring))
        _BALL_IMG_CACHE[key] = img
    return img


def draw_sphere(cv, cx, cy, r, text, base, light, tcolor="#ffffff", font=FONT_NUM, ring=None):
    """在 Canvas 上画一颗 3D 立体球(投影 + 径向渐变 + 文字 + 可选命中金环)。"""
    cv.create_oval(cx - r + 2, cy - r + 3, cx + r + 2, cy + r + 3, fill=SHADOW, outline="")
    cv.create_oval(cx - r, cy - r, cx + r, cy + r, fill=base,
                   outline=_blend(base, SHADOW, 0.45))
    layers = 5
    for i in range(1, layers + 1):
        t = i / (layers + 1)
        col = _blend(base, light, t)
        rr = r * (1 - i * 0.13)
        ox, oy = -i * 0.5, -i * 0.75
        cv.create_oval(cx - rr + ox, cy - rr + oy, cx + rr + ox, cy + rr + oy,
                       fill=col, outline="")
    cv.create_text(cx, cy + 1, text=text, fill=tcolor, font=font)
    if ring:  # 命中标记: 金色描边环
        cv.create_oval(cx - r, cy - r, cx + r, cy + r, outline=ring, width=2)


# 波色调色板: (base, light)
WAVE_COLORS = {
    "红波": ("#c0392b", "#ff7878"),
    "绿波": ("#1f7a32", "#3fb958"),
    "蓝波": ("#1f5fa8", "#5fa8e0"),
}
HIT_RING = "#f1c40f"  # 命中金环(叠加在波色球上)

# 球样式: (base, light, r, font) — base/light 为非号码球(生肖等)的默认色;
# 号码球会按所属波色覆盖 base/light
BALL_STYLES = {
    "reg": ("#a83a3a", "#e57373", 15, FONT_NUM_S),
    "special": ("#c0392b", "#ff7878", 17, FONT_NUM),
    "pool": ("#9c3535", "#dd6868", 14, FONT_NUM_S),
    "wide": ("#8f3030", "#d06060", 13, FONT_NUM_S),
    "zodiac": ("#a83a3a", "#e57373", 18, FONT_ZOD),
    "muted": ("#3a3a48", "#5a5a6a", 15, FONT_NUM_S),
    "reg_l": ("#a83a3a", "#e57373", 19, FONT_NUM),                     # 开奖记录页: 加大平码(+2mm)
    "special_l": ("#c0392b", "#ff7878", 21, ("Consolas", 13, "bold")),  # 开奖记录页: 加大特码(+2mm)
}


def make_ball(parent, text, kind="reg", hit=False):
    """创建并返回一颗球的 Canvas(不布局), 供 grid/pack 自由放置。

    号码球按所属波色渲染; hit=True 叠加金色命中环。
    优先用 PIL 超采样渲染(边缘细腻/圆润/高清), 无 PIL 时降级为矢量绘制。"""
    base, light, r, font = BALL_STYLES[kind]
    try:
        n = int(text)
    except ValueError:
        n = None
    if n is not None and 1 <= n <= 49:  # 号码球: 按波色上色
        wc = WAVE_COLORS.get(wave_of_number(n))
        if wc:
            base, light = wc
    ring = HIT_RING if hit else None
    size = r * 2 + 6
    cv = tk.Canvas(parent, width=size, height=size, bg=parent["bg"], highlightthickness=0)
    if _PIL_OK:
        img = _get_ball_image(size, r, base, light, ring)
        cv.create_image(size / 2, size / 2, image=img, anchor="center")
        cv.image = img  # 防 GC
        cv.create_text(size / 2, size / 2 + 1, text=text, fill="#ffffff", font=font)
    else:
        draw_sphere(cv, size / 2, size / 2, r, text, base, light, "#ffffff", font, ring=ring)
    return cv


def add_ball(parent, text, kind="reg", hit=False):
    """添加一颗球(左对齐 pack)。号码球按所属波色渲染; hit=True 叠加金色命中环。"""
    cv = make_ball(parent, text, kind, hit)
    cv.pack(side="left", padx=3, pady=2)
    return cv


class ScrollArea:
    """可滚动的卡片流容器。"""

    def __init__(self, parent):
        self.canvas = tk.Canvas(parent, bg=BG_MAIN, highlightthickness=0, bd=0)
        self.canvas.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(parent, orient="vertical", command=self.canvas.yview)
        sb.pack(side="right", fill="y")
        self.canvas.configure(yscrollcommand=sb.set)
        self.inner = tk.Frame(self.canvas, bg=BG_MAIN)
        self.inner.bind("<Configure>",
                        lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.inner, anchor="nw", tags="inner")
        self.canvas.bind("<Configure>", self._on_resize)

    def _on_resize(self, e):
        self.canvas.itemconfigure("inner", width=e.width)

    def clear(self):
        for w in self.inner.winfo_children():
            w.destroy()


class App:
    def __init__(self, root):
        self.root = root
        self.queue = queue.Queue()
        self.results = None
        self.worker = None
        root.title("澳门六合彩开奖记录分析与预测-猎手2026版")
        root.geometry("1040x740")
        root.configure(bg=BG_MAIN)
        self._setup_style()
        self._build_ui()
        self._poll()

    def _setup_style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TNotebook", background=BG_MAIN, borderwidth=0, tabmargins=(6, 6, 6, 0))
        style.configure("TNotebook.Tab", background=BG_CARD2, foreground=MUTED,
                        padding=(20, 9), font=FONT, borderwidth=0)
        style.map("TNotebook.Tab",
                  background=[("selected", BG_CARD)],
                  foreground=[("selected", ACCENT)],
                  expand=[("selected", (1, 1, 1, 0))])
        style.configure("TCheckbutton", background=BG_MAIN, foreground=TEXT, font=FONT)
        style.configure("TSpinbox", fieldbackground=BG_CARD2, foreground=TEXT,
                        background=BG_MAIN, arrowcolor=PRIMARY, bordercolor=BORDER,
                        lightcolor=BORDER, darkcolor=BORDER, insertcolor=PRIMARY)
        style.configure("Horizontal.TProgressbar", background=ACCENT,
                        troughcolor=BG_CARD2, borderwidth=0, lightcolor=ACCENT, darkcolor=ACCENT)
        style.configure("Vertical.TScrollbar", background=BG_CARD2,
                        troughcolor=BG_MAIN, borderwidth=0, arrowcolor=PRIMARY,
                        gripcount=0)
        style.configure("Treeview", background=BG_CARD, fieldbackground=BG_CARD,
                        foreground=TEXT, borderwidth=0, rowheight=24, font=FONT_SM)
        style.configure("Treeview.Heading", background=BG_CARD2, foreground=MUTED,
                        font=FONT_SM, borderwidth=0, relief="flat")
        style.map("Treeview", background=[("selected", PRIMARY_L)],
                  foreground=[("selected", TEXT)])
        style.map("Treeview.Heading", background=[("active", BG_CARD2)])

    # ---------------- UI 构建 ----------------
    def _build_ui(self):
        # 顶部标题区
        header = tk.Frame(self.root, bg=BG_MAIN)
        header.pack(fill="x", padx=20, pady=(16, 2))
        tk.Label(header, text="澳门六合彩", fg=ACCENT, bg=BG_MAIN,
                 font=FONT_TITLE).pack(side="left")
        tk.Label(header, text="  开奖记录分析与预测-猎手2026版", fg=TEXT, bg=BG_MAIN,
                 font=("Microsoft YaHei UI", 13)).pack(side="left", padx=2)
        tk.Label(header, text="【AI大模型反推理 + 数学模型 + 马尔可夫模型 + 贝叶斯后验】", fg=MUTED, bg=BG_MAIN,
                 font=FONT_SM).pack(side="left", padx=(10, 0), pady=(4, 0))
        accent = tk.Frame(self.root, bg=PRIMARY_D, height=2)
        accent.pack(fill="x", padx=20, pady=(0, 6))
        # 渐变细线
        glow = tk.Frame(self.root, bg=BORDER, height=1)
        glow.pack(fill="x", padx=20, pady=(0, 6))

        # 即将开奖期次 醒目横幅(红底白字 + 实时倒计时)
        self._next_issue = None
        self._next_open_dt = None
        nbanner = tk.Frame(self.root, bg=ACCENT)
        nbanner.pack(fill="x", padx=20, pady=(2, 6))
        tk.Label(nbanner, text="⏰  即将开奖", bg=ACCENT, fg="#ffffff",
                 font=("Microsoft YaHei UI", 11, "bold")).pack(
                     side="left", padx=(14, 4), pady=7)
        self.next_expect_lbl = tk.Label(nbanner, text="第 - 期", bg=ACCENT, fg="#ffffff",
                                        font=("Microsoft YaHei UI", 16, "bold"))
        self.next_expect_lbl.pack(side="left", padx=6)
        self.next_countdown_lbl = tk.Label(nbanner, text="", bg=ACCENT, fg="#ffe8e8",
                                          font=("Microsoft YaHei UI", 11, "bold"))
        self.next_countdown_lbl.pack(side="right", padx=14)
        self._set_next_issue(load_cache())
        self._tick_countdown()

        # 控制栏
        bar = tk.Frame(self.root, bg=BG_MAIN)
        bar.pack(fill="x", padx=20, pady=4)
        self.run_btn = tk.Button(bar, text="▶  开始分析", command=self.start,
                                 bg=ACCENT, fg="#ffffff", activebackground=PRIMARY,
                                 activeforeground="#ffffff", relief="flat", bd=0,
                                 font=("Microsoft YaHei UI", 10, "bold"),
                                 padx=18, pady=6, cursor="hand2")
        self.run_btn.pack(side="left")
        self._hover(self.run_btn, ACCENT, PRIMARY)

        self.entry_btn = tk.Button(bar, text="✎  录入开奖", command=self._manual_entry,
                                   bg=BG_CARD2, fg=PRIMARY_D, activebackground=PRIMARY_L,
                                   activeforeground="#ffffff", relief="flat", bd=0,
                                   font=("Microsoft YaHei UI", 10, "bold"),
                                   padx=14, pady=6, cursor="hand2")
        self.entry_btn.pack(side="right")
        self._hover(self.entry_btn, BG_CARD2, PRIMARY_L)

        tk.Label(bar, text="  回测期数", bg=BG_MAIN, fg=MUTED, font=FONT).pack(side="left", padx=(14, 4))
        self.bt_var = tk.IntVar(value=60)
        ttk.Spinbox(bar, from_=5, to=100, textvariable=self.bt_var, width=5, font=FONT).pack(side="left")
        self.refresh_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(bar, text="强制刷新数据", variable=self.refresh_var).pack(side="left", padx=12)
        self.prog = ttk.Progressbar(bar, mode="indeterminate", length=130)
        self.prog.pack(side="left", padx=12)
        self.status = tk.Label(bar, text="● 就绪。点击「开始分析」", bg=BG_MAIN, fg=MUTED, font=FONT)
        self.status.pack(side="left", padx=10)

        # 标签页
        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill="both", expand=True, padx=12, pady=(6, 12))
        self.areas = {}
        titles = [("overview", "总览"), ("full", "三组完整预测"),
                  ("single", "五组单颗特码"), ("dims", "六维度属性"),
                  ("pools", "号码集合"), ("zodiac", "三生肖"), ("zodiac_quad", "四生肖"), ("zodiac_six", "六生肖")]
        for key, title in titles:
            page = tk.Frame(self.nb, bg=BG_MAIN)
            self.nb.add(page, text=title)
            self.areas[key] = ScrollArea(page)
        self._scroll_canvases = {a.canvas for a in self.areas.values()}
        # 开奖记录页(列表 + 详情 自定义布局)
        self._records_page = tk.Frame(self.nb, bg=BG_MAIN)
        self.nb.add(self._records_page, text="开奖记录")
        self._build_records_page(self._records_page)
        # 历史预测页(列表 + 详情 自定义布局)
        self._hist_page = tk.Frame(self.nb, bg=BG_MAIN)
        self.nb.add(self._hist_page, text="历史预测")
        self._build_history_page(self._hist_page)
        # 全局鼠标滚轮: 路由到指针所在的可滚动区(支持悬停在卡片/子控件上)
        self.root.bind_all("<MouseWheel>", self._on_wheel)
        self.root.bind_all("<Button-4>", self._on_wheel)
        self.root.bind_all("<Button-5>", self._on_wheel)
        self.nb.bind("<<NotebookTabChanged>>", self._on_tab_changed)
        self._render_placeholder()

    def _hover(self, btn, normal, hover):
        btn.bind("<Enter>", lambda e: btn.config(bg=hover))
        btn.bind("<Leave>", lambda e: btn.config(bg=normal))

    def _render_placeholder(self):
        a = self.areas["overview"]
        a.clear()
        card = self._card(a.inner, "欢迎使用")
        tk.Label(card, text="点击右上方「▶ 开始分析」启动预测。\n\n程序自动抓取 macaujc.com 历史开奖, "
                            "采用统计模型 + 大模型反推理, 给出多维度预测。\n\n" + DISCLAIMER,
                 bg=BG_CARD, fg=TEXT, font=FONT, justify="left").pack(anchor="w", pady=6)

    # ---------------- 即将开奖期次 ----------------
    def _set_next_issue(self, records):
        """根据记录更新"即将开奖"横幅(期号 + 倒计时目标)。"""
        ni = compute_next_issue(records)
        self._next_issue = ni
        if ni:
            self.next_expect_lbl.config(text=f"第 {ni['expect']} 期")
            self._next_open_dt = self._parse_dt(ni.get("open_time", ""))
        else:
            self.next_expect_lbl.config(text="第 - 期")
            self._next_open_dt = None

    @staticmethod
    def _parse_dt(s):
        if not s:
            return None
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
            try:
                return datetime.strptime(s, fmt)
            except ValueError:
                continue
        return None

    def _tick_countdown(self):
        """每秒刷新距开奖倒计时; 已过点则提示等待结果。"""
        dt = self._next_open_dt
        if dt:
            remain = (dt - datetime.now()).total_seconds()
            if remain > 0:
                h = int(remain // 3600)
                m = int((remain % 3600) // 60)
                s = int(remain % 60)
                mmdd = f"{dt.month:02d}-{dt.day:02d} {dt.hour:02d}:{dt.minute:02d}"
                self.next_countdown_lbl.config(
                    text=f"距 {mmdd} 开奖  还有 {h:02d}:{m:02d}:{s:02d}")
            else:
                self.next_countdown_lbl.config(text="开奖进行中 · 等待结果")
        else:
            self.next_countdown_lbl.config(text="")
        self.root.after(1000, self._tick_countdown)

    # ---------------- 卡片构件 ----------------
    def _card(self, parent, title="", star=False, accent=PRIMARY, expect=""):
        """带投影 + 顶部彩条的立体卡片, 返回内容 Frame。

        expect 非空时在标题后追加期次标签(如【第2026196期】), 置于"★ 推荐"之前。
        """
        shadow = tk.Frame(parent, bg=SHADOW)
        shadow.pack(fill="x", padx=14, pady=(8, 12))
        card = tk.Frame(shadow, bg=BG_CARD)
        card.pack(fill="x", padx=(0, 4), pady=(0, 5))
        tk.Frame(card, bg=accent, height=3).pack(fill="x")
        if title:
            hd = tk.Frame(card, bg=BG_CARD)
            hd.pack(fill="x", padx=16, pady=(10, 4))
            tk.Label(hd, text=title, bg=BG_CARD, fg=accent, font=FONT_H).pack(side="left")
            if expect:
                tk.Label(hd, text=f"【第{expect}期】", bg=BG_CARD, fg=TEXT2,
                         font=FONT_SM).pack(side="left", padx=(6, 0))
            if star:
                tk.Label(hd, text="★ 推荐", bg=BG_CARD, fg=ACCENT, font=FONT).pack(side="left", padx=8)
        body = tk.Frame(card, bg=BG_CARD)
        body.pack(fill="x", padx=16, pady=(2, 12))
        return body

    def _metric_tile(self, parent, label, value, color=PRIMARY_L):
        tile = tk.Frame(parent, bg=BG_CARD2)
        tile.pack(side="left", padx=5)
        tk.Frame(tile, bg=color, height=2).pack(fill="x")
        inner = tk.Frame(tile, bg=BG_CARD2)
        inner.pack(padx=16, pady=(8, 10))
        tk.Label(inner, text=value, bg=BG_CARD2, fg=color,
                 font=("Consolas", 16, "bold")).pack()
        tk.Label(inner, text=label, bg=BG_CARD2, fg=MUTED, font=FONT_SM).pack()
        return tile

    def _metric_row(self, parent, pairs):
        row = tk.Frame(parent, bg=BG_CARD)
        row.pack(fill="x", pady=(4, 8))
        for label, value, color in pairs:
            cell = tk.Frame(row, bg=BG_CARD)
            cell.pack(side="left", expand=True, fill="x")
            tk.Label(cell, text=value, bg=BG_CARD, fg=color,
                     font=("Consolas", 13, "bold")).pack(anchor="w")
            tk.Label(cell, text=label, bg=BG_CARD, fg=MUTED, font=FONT_SM).pack(anchor="w")

    def _lift_color(self, lift):
        if lift > 1e-9:
            return OK
        if lift < -1e-9:
            return WARN
        return MUTED

    def _sig_verdict(self, lift, std_error):
        """提升度相对标准误的显著性: 仅当 lift > std_error 才算"有效提升",
        否则视为随机波动(噪声)。返回 (short, full, color)。

        - lift >  se : ✓ 有效提升(绿)   -- 提升大于标准误, 统计上可信
        - |lift|<= se: ≈ 噪声范围(灰)   -- 提升在标准误以内, 与随机无异
        - lift < -se : ✗ 低于基线(橙)   -- 显著低于随机基线
        - se<=0      : 数据不足(灰)
        """
        try:
            se = float(std_error or 0.0)
        except (TypeError, ValueError):
            se = 0.0
        if se <= 1e-9:
            return ("数据不足", "数据不足", TEXT2)
        if lift > se:
            return ("✓有效", "✓ 有效提升", OK)
        if lift < -se:
            return ("✗负向", "✗ 低于基线", WARN)
        return ("≈噪声", "≈ 噪声范围", TEXT2)

    def _significance_chip(self, parent, lift, std_error):
        """提升显著性色块: 一眼判断"是否真有提升"。
        绿=lift>标准误(有效), 灰=|lift|<=标准误(噪声), 橙=显著低于基线。"""
        _short, full, color = self._sig_verdict(lift, std_error)
        try:
            se = float(std_error or 0.0)
        except (TypeError, ValueError):
            se = 0.0
        chip = tk.Frame(parent, bg=color)
        chip.pack(anchor="w", pady=(4, 2))
        tk.Label(chip,
                 text=f" 提升显著性  {full}   提升 {lift:+.1%}   标准误 {se:.1%} ",
                 bg=color, fg="#ffffff",
                 font=("Microsoft YaHei UI", 9, "bold")).pack(padx=2, pady=2)
        return chip

    # ---------------- 运行 ----------------
    def start(self):
        if self.worker and self.worker.is_alive():
            return
        self.run_btn.config(state="disabled")
        self.prog.start(12)
        self._status("运行中...", ACCENT)
        self.worker = threading.Thread(target=self._work, daemon=True)
        self.worker.start()

    def _work(self):
        try:
            self._msg("加载历史数据...")
            records = load_history(HISTORY_YEARS, refresh=self.refresh_var.get())
            if len(records) < 30:
                self.queue.put(("error", f"数据不足({len(records)}期), 请检查网络或勾选强制刷新。"))
                return
            self._msg(f"已加载 {len(records)} 期, 统计分析中...")
            report = build_report(records)
            self._msg("大模型反推理(可选)...")
            cfg = load_config()
            llm_result = None
            if cfg:
                try:
                    recent = [r.special for r in records[-10:]]
                    llm_result = reason(summarize_for_llm(report, 10), recent, cfg)
                    self._msg("大模型完成, 预测中...")
                except Exception as e:
                    self._msg(f"大模型失败, 降级: {e}")
            else:
                self._msg("未配置 LLM, 仅用统计层...")
            bt = self.bt_var.get()
            self._msg("三组完整预测 + 回测...")
            groups = predict_all(records, llm_result, bt)
            self._msg("五组特码预测...")
            sp = predict_special_groups(records, llm_result, bt)
            self._msg("六维度预测...")
            dims = predict_dimensions(records, llm_result, bt)
            self._msg("号码集合预测...")
            pools = predict_special_pools(records, llm_result, bt)
            wide = predict_wide_pool(records, llm_result, bt)
            self._msg("三生肖预测...")
            zp = predict_zodiac_pool(records, llm_result, bt)
            self._msg("四生肖预测...")
            zq = predict_zodiac_quad(records, llm_result, bt)
            self._msg("六生肖预测...")
            zs = predict_zodiac_six(records, llm_result, bt)
            self.queue.put(("done", dict(records=records, report=report, llm=llm_result,
                                         groups=groups, sp=sp, dims=dims,
                                         pools=pools, wide=wide, zp=zp, zq=zq, zs=zs, bt=bt)))
        except Exception as e:
            self.queue.put(("error", f"{type(e).__name__}: {e}"))

    def _msg(self, text):
        self.queue.put(("msg", text))

    def _status(self, text, color=MUTED):
        self.status.config(text="● " + text, fg=color)

    def _on_wheel(self, e):
        """全局鼠标滚轮: 滚动指针当前所在的滚动区(冒泡到所属 Canvas)。"""
        w = self.root.winfo_containing(e.x_root, e.y_root)
        cv = None
        while w:
            if isinstance(w, tk.Canvas) and w in self._scroll_canvases:
                cv = w
                break
            try:
                w = w.master
            except Exception:
                break
        if cv is None:
            return
        if e.delta != 0:
            direction = -1 if e.delta > 0 else 1
            px = (abs(e.delta) / 120) * 100
        elif getattr(e, "num", 0) == 4:
            direction, px = -1, 100
        elif getattr(e, "num", 0) == 5:
            direction, px = 1, 100
        else:
            return
        self._scroll_canvas(cv, direction, px)

    @staticmethod
    def _scroll_canvas(cv, direction, px):
        """按像素步进滚动指定 Canvas; 返回是否实际滚动。"""
        first, last = cv.yview()
        if last - first >= 1.0:
            return False
        bbox = cv.bbox("all")
        if not bbox:
            return False
        content_h = bbox[3] - bbox[1]
        if content_h <= 0:
            return False
        cv.yview_moveto(first + direction * px / content_h)
        return True

    def _poll(self):
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                if kind == "msg":
                    self._status(payload, ACCENT)
                elif kind == "error":
                    self.prog.stop()
                    self.run_btn.config(state="normal")
                    self._status("错误", WARN)
                    messagebox.showerror("错误", payload)
                elif kind == "done":
                    self.results = payload
                    self._set_next_issue(payload["records"])
                    # 保存本期预测到历史档案
                    saved = False
                    try:
                        saved = save_run(serialize_run(payload))
                    except Exception:
                        saved = False
                    self._render_all()
                    self.prog.stop()
                    self.run_btn.config(state="normal")
                    self._status("完成 · 已存档" if saved else "完成 · 存档失败",
                                 OK if saved else WARN)
                    self.nb.select(self.areas["overview"].canvas.master)
                elif kind == "hist_verified":
                    self._hist_fetching = False
                    if payload is not None:
                        self._hist_records = payload
                        self._set_next_issue(payload)
                        self._history_refresh()
                        self._status("已获取最新开奖", OK)
                    else:
                        self._status("获取最新开奖失败", WARN)
                        self.hist_status.config(text="获取最新开奖失败, 显示缓存数据")
        except queue.Empty:
            pass
        self.root.after(200, self._poll)

    # ---------------- 渲染 ----------------
    def _render_all(self):
        r = self.results
        recs = r["records"]
        report = r["report"]
        last = recs[-1]
        self._render_overview(recs, report, last, r["llm"])
        self._render_full(r["groups"])
        self._render_single(r["sp"])
        self._render_dims(r["dims"])
        self._render_pools(r["pools"], r["wide"])
        self._render_zodiac(r["zp"])
        self._render_zodiac_quad(r["zq"])
        self._render_zodiac_six(r["zs"])

    def _render_overview(self, recs, report, last, llm):
        a = self.areas["overview"]
        a.clear()
        # 指标磁贴
        tile_card = self._card(a.inner, "数据概览")
        trow = tk.Frame(tile_card, bg=BG_CARD)
        trow.pack(fill="x", pady=4)
        self._metric_tile(trow, "历史期数", f"{len(recs)}")
        self._metric_tile(trow, "最近期号", f"{last.expect}")
        self._metric_tile(trow, "最近特码", f"{last.special:02d}", ACCENT)
        days = (recs[-1].open_time[:10] >= recs[0].open_time[:10]) and (len(recs))
        tk.Label(tile_card, text=f"时间范围: {recs[0].open_time[:10]}  ~  {last.open_time[:10]}",
                 bg=BG_CARD, fg=MUTED, font=FONT).pack(anchor="w", pady=(6, 2))
        # 反推理模型
        mcard = self._card(a.inner, "反推理候选数学模型")
        for name, score, detail in report["models"]:
            row = tk.Frame(mcard, bg=BG_CARD)
            row.pack(fill="x", pady=2)
            tk.Label(row, text=name, bg=BG_CARD, fg=TEXT, font=FONT, width=18, anchor="w").pack(side="left")
            bar = tk.Canvas(row, width=120, height=12, bg=BG_CARD, highlightthickness=0)
            bar.pack(side="left", padx=6)
            bar.create_rectangle(0, 2, 120, 10, fill=BG_CARD2, outline="")
            bar.create_rectangle(0, 2, int(120 * score), 10, fill=PRIMARY, outline="")
            tk.Label(row, text=f"{score:.3f}", bg=BG_CARD, fg=PRIMARY_L,
                     font=("Consolas", 10, "bold")).pack(side="left", padx=6)
            tk.Label(row, text=detail, bg=BG_CARD, fg=MUTED, font=FONT_SM).pack(side="left")
        # 大模型
        lcard = self._card(a.inner, "大模型反推理")
        if llm:
            tk.Label(lcard, text="模型清单: " + (", ".join(llm.inferred_models) or "(未解析)"),
                     bg=BG_CARD, fg=PRIMARY_L, font=FONT, wraplength=880, justify="left",
                     anchor="w").pack(fill="x", pady=2)
            tk.Label(lcard, text="推理: " + llm.reasoning[:300], bg=BG_CARD, fg=MUTED,
                     font=FONT_SM, wraplength=880, justify="left", anchor="w").pack(fill="x")
        else:
            tk.Label(lcard, text="未启用(编辑 config.ini 填入 api_key)", bg=BG_CARD,
                     fg=MUTED, font=FONT).pack(anchor="w")
        # 免责
        dcard = self._card(a.inner, "科学提示", accent=MUTED)
        tk.Label(dcard, text=DISCLAIMER, bg=BG_CARD, fg=MUTED, font=FONT_SM,
                 justify="left", anchor="w").pack(fill="x")

    def _render_full(self, groups):
        a = self.areas["full"]
        a.clear()
        for i, g in enumerate(groups):
            body = self._card(a.inner, f"{g.name}组  三组完整预测", star=(i == 0))
            # 平码球
            brow = tk.Frame(body, bg=BG_CARD)
            brow.pack(fill="x", pady=(2, 4))
            tk.Label(brow, text="平码", bg=BG_CARD, fg=MUTED, font=FONT_SM, width=4).pack(side="left")
            for n in g.regular:
                add_ball(brow, f"{n:02d}", "reg")
            # 特码球
            srow = tk.Frame(body, bg=BG_CARD)
            srow.pack(fill="x", pady=(4, 6))
            tk.Label(srow, text="特码", bg=BG_CARD, fg=MUTED, font=FONT_SM, width=4).pack(side="left")
            add_ball(srow, f"{g.special:02d}", "special")
            tk.Label(srow, text=f"  {g.strategy}", bg=BG_CARD, fg=MUTED, font=FONT_SM).pack(side="left", padx=8)
            # 指标
            self._metric_row(body, [
                ("特码命中率", f"{g.backtest_special_hit:.1%}", PRIMARY_L),
                ("平均平码命中", f"{g.backtest_regular_hits:.2f}/6", TEXT),
            ])

    def _render_single(self, sp):
        a = self.areas["single"]
        a.clear()
        for i, g in enumerate(sp):
            body = self._card(a.inner, f"特码 {g.name}组", star=(i == 0))
            row = tk.Frame(body, bg=BG_CARD)
            row.pack(fill="x", pady=(2, 6))
            add_ball(row, f"{g.special:02d}", "special")
            tk.Label(row, text=f"  {g.strategy}", bg=BG_CARD, fg=MUTED, font=FONT_SM).pack(side="left", padx=6)
            self._metric_row(body, [
                ("命中率", f"{g.backtest_hit:.1%}", PRIMARY_L),
                ("提升度", f"{g.lift:+.1%}", self._lift_color(g.lift)),
                ("标准误", f"{g.std_error:.1%}", TEXT),
                ("稳定性", f"{g.stability:.2f}", TEXT),
            ])
            self._significance_chip(body, g.lift, g.std_error)

    def _render_dims(self, dims):
        a = self.areas["dims"]
        a.clear()
        body = self._card(a.inner, "六维度特码属性  (各维独立贝叶斯预测 · 精确组合基线)")
        joint_n = dims[0].joint_number if dims else 0
        if joint_n:
            jrow = tk.Frame(body, bg=BG_CARD)
            jrow.pack(fill="x", pady=(2, 6))
            tk.Label(jrow, text=f"最一致参考特码: {joint_n:02d}", bg=BG_CARD, fg=ACCENT,
                     font=("Consolas", 13, "bold")).pack(side="left")
            tk.Label(jrow, text="  ← 与六维预测最一致的号码(展示用, 不反向决定维度)",
                     bg=BG_CARD, fg=MUTED, font=FONT_SM).pack(side="left", padx=4)
        # 表头
        head = tk.Frame(body, bg=BG_CARD)
        head.pack(fill="x", pady=(2, 4))
        for j, h in enumerate(["维度", "预测值", "准确率", "随机基线", "提升度", "标准误", "稳定性", "显著性"]):
            tk.Label(head, text=h, bg=BG_CARD, fg=MUTED, font=FONT_SM,
                     width=10 if 1 < j < 7 else 8, anchor="w").pack(side="left")
        for d in dims:
            row = tk.Frame(body, bg=BG_CARD)
            row.pack(fill="x", pady=2)
            sig_short, _sig_full, sig_color = self._sig_verdict(d.lift, d.std_error)
            vals = [d.name, d.value, f"{d.accuracy:.1%}", f"{d.baseline:.1%}",
                    f"{d.lift:+.1%}", f"{d.std_error:.1%}", f"{d.stability:.2f}", sig_short]
            colors = [TEXT, PRIMARY_L, PRIMARY_L, MUTED, self._lift_color(d.lift), TEXT, TEXT, sig_color]
            widths = [8, 8, 10, 10, 10, 10, 10, 9]
            for v, c, w in zip(vals, colors, widths):
                tk.Label(row, text=v, bg=BG_CARD, fg=c, font=FONT, width=w, anchor="w").pack(side="left")
        tk.Label(body, text="提升度 > 0 表示优于随机基线; 显著性: ✓有效=lift>标准误(可信), "
                            "≈噪声=|lift|≤标准误(与随机无异), ✗负=显著低于基线",
                 bg=BG_CARD, fg=MUTED, font=FONT_SM).pack(anchor="w", pady=(8, 0))
        # 历史众数参考(全期最频, 不参与主预测; 主预测值=近期加权, 会随近期变动)
        tk.Label(body, text="各维度历史众数(独立统计, 可能相互冲突, 仅参考):  "
                            + "  ".join(f"{d.name} {d.mode_value or '-'}" for d in dims),
                 bg=BG_CARD, fg=MUTED, font=FONT_SM).pack(anchor="w", pady=(6, 0))

    def _render_pools(self, pools, wide):
        a = self.areas["pools"]
        a.clear()
        tk.Label(a.inner, text="特码号码集合  (命中 = 真实特码落在集合内)", bg=BG_MAIN,
                 fg=MUTED, font=FONT_SM).pack(anchor="w", padx=18, pady=(6, 0))
        for i, p in enumerate(pools):
            body = self._card(a.inner, f"集合 {p.name}组  ({len(p.numbers)}颗)", star=(i == 0))
            row = tk.Frame(body, bg=BG_CARD)
            row.pack(fill="x", pady=(2, 6))
            for n in p.numbers:
                add_ball(row, f"{n:02d}", "pool")
            self._metric_row(body, [
                ("命中率", f"{p.hit_rate:.1%}", PRIMARY_L),
                ("基线", f"{p.baseline:.1%}", MUTED),
                ("提升度", f"{p.lift:+.1%}", self._lift_color(p.lift)),
                ("稳定性", f"{p.stability:.2f}", TEXT),
            ])
            self._significance_chip(body, p.lift, p.std_error)
        # 20 颗大集合
        body = self._card(a.inner, f"20 颗特码大集合  (单组 · 多信号融合)", star=True, accent=ACCENT)
        row = tk.Frame(body, bg=BG_CARD)
        row.pack(fill="x", pady=(2, 6))
        for n in wide.numbers:
            add_ball(row, f"{n:02d}", "wide")
        self._metric_row(body, [
            ("命中率", f"{wide.hit_rate:.1%}", PRIMARY_L),
            ("基线", f"{wide.baseline:.1%}", MUTED),
            ("提升度", f"{wide.lift:+.1%}", self._lift_color(wide.lift)),
            ("稳定性", f"{wide.stability:.2f}", TEXT),
        ])
        self._significance_chip(body, wide.lift, wide.std_error)

    def _render_zodiac(self, zp):
        a = self.areas["zodiac"]
        a.clear()
        body = self._card(a.inner, "特码三生肖  (单组)", star=True, accent=ACCENT)
        tk.Label(body, text="命中 = 真实特码生肖落在 3 个内    随机基线 = 3/12 = 25%",
                 bg=BG_CARD, fg=MUTED, font=FONT_SM).pack(anchor="w", pady=(2, 8))
        row = tk.Frame(body, bg=BG_CARD)
        row.pack(fill="x", pady=(2, 8))
        for z in zp.zodiacs:
            add_ball(row, z, "zodiac")
        self._metric_row(body, [
            ("命中率", f"{zp.hit_rate:.1%}", PRIMARY_L),
            ("基线", f"{zp.baseline:.1%}", MUTED),
            ("提升度", f"{zp.lift:+.1%}", self._lift_color(zp.lift)),
            ("标准误", f"{zp.std_error:.1%}", TEXT),
            ("稳定性", f"{zp.stability:.2f}", TEXT),
        ])
        self._significance_chip(body, zp.lift, zp.std_error)
        dcard = self._card(a.inner, "科学提示", accent=MUTED)
        tk.Label(dcard, text=DISCLAIMER, bg=BG_CARD, fg=MUTED, font=FONT_SM,
                 justify="left", anchor="w").pack(fill="x")

    def _render_zodiac_quad(self, zq):
        a = self.areas["zodiac_quad"]
        a.clear()
        body = self._card(a.inner, "特码四生肖  (单组)", star=True, accent=ACCENT)
        tk.Label(body, text="命中 = 真实特码生肖落在 4 个内    随机基线 = 4/12 ≈ 33.3%",
                 bg=BG_CARD, fg=MUTED, font=FONT_SM).pack(anchor="w", pady=(2, 8))
        row = tk.Frame(body, bg=BG_CARD)
        row.pack(fill="x", pady=(2, 8))
        for z in zq.zodiacs:
            add_ball(row, z, "zodiac")
        self._metric_row(body, [
            ("命中率", f"{zq.hit_rate:.1%}", PRIMARY_L),
            ("基线", f"{zq.baseline:.1%}", MUTED),
            ("提升度", f"{zq.lift:+.1%}", self._lift_color(zq.lift)),
            ("标准误", f"{zq.std_error:.1%}", TEXT),
            ("稳定性", f"{zq.stability:.2f}", TEXT),
        ])
        self._significance_chip(body, zq.lift, zq.std_error)
        dcard = self._card(a.inner, "科学提示", accent=MUTED)
        tk.Label(dcard, text=DISCLAIMER, bg=BG_CARD, fg=MUTED, font=FONT_SM,
                 justify="left", anchor="w").pack(fill="x")

    def _render_zodiac_six(self, zs):
        a = self.areas["zodiac_six"]
        a.clear()
        body = self._card(a.inner, "特码六生肖  (单组)", star=True, accent=ACCENT)
        tk.Label(body, text="命中 = 真实特码生肖落在 6 个内    随机基线 = 6/12 = 50%",
                 bg=BG_CARD, fg=MUTED, font=FONT_SM).pack(anchor="w", pady=(2, 8))
        row = tk.Frame(body, bg=BG_CARD)
        row.pack(fill="x", pady=(2, 8))
        for z in zs.zodiacs:
            add_ball(row, z, "zodiac")
        self._metric_row(body, [
            ("命中率", f"{zs.hit_rate:.1%}", PRIMARY_L),
            ("基线", f"{zs.baseline:.1%}", MUTED),
            ("提升度", f"{zs.lift:+.1%}", self._lift_color(zs.lift)),
            ("标准误", f"{zs.std_error:.1%}", TEXT),
            ("稳定性", f"{zs.stability:.2f}", TEXT),
        ])
        self._significance_chip(body, zs.lift, zs.std_error)
        dcard = self._card(a.inner, "科学提示", accent=MUTED)
        tk.Label(dcard, text=DISCLAIMER, bg=BG_CARD, fg=MUTED, font=FONT_SM,
                 justify="left", anchor="w").pack(fill="x")

    # ---------------- 开奖记录页 ----------------
    def _build_records_page(self, page):
        """开奖记录浏览: 左侧列表(期号/时间/平码/特码) + 右侧详情(波色球+生肖+统计)。"""
        bar = tk.Frame(page, bg=BG_MAIN)
        bar.pack(fill="x", padx=12, pady=(8, 4))
        tk.Label(bar, text="期号筛选", bg=BG_MAIN, fg=MUTED, font=FONT_SM).pack(side="left")
        self.rec_filter_var = tk.StringVar()
        ent = tk.Entry(bar, textvariable=self.rec_filter_var, width=14, font=FONT,
                       relief="solid", borderwidth=1, bg=BG_CARD, fg=TEXT, insertbackground=TEXT)
        ent.pack(side="left", padx=6)
        self.rec_filter_var.trace_add("write", lambda *a: self._records_refresh())
        tk.Button(bar, text="⟳ 刷新", command=lambda: self._records_refresh(reload=True),
                  bg=BG_CARD2, fg=TEXT, relief="flat", bd=0, font=FONT_SM,
                  padx=10, pady=3, cursor="hand2").pack(side="left", padx=6)
        self.rec_status = tk.Label(bar, text="", bg=BG_MAIN, fg=MUTED, font=FONT_SM)
        self.rec_status.pack(side="left", padx=14)

        paned = tk.PanedWindow(page, orient="horizontal", bg=BG_MAIN,
                               sashwidth=6, sashrelief="flat")
        paned.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        left = tk.Frame(paned, bg=BG_MAIN)
        paned.add(left, minsize=300, width=420)
        cols = ("expect", "time", "regular", "special")
        headers = {"expect": "期号", "time": "开奖时间", "regular": "平码", "special": "特码"}
        self.rec_tree = ttk.Treeview(left, columns=cols, show="headings", height=24)
        for c in cols:
            self.rec_tree.heading(c, text=headers[c])
        self.rec_tree.column("expect", width=85, anchor="w")
        self.rec_tree.column("time", width=140, anchor="w")
        self.rec_tree.column("regular", width=175, anchor="w")
        self.rec_tree.column("special", width=50, anchor="center")
        sb = ttk.Scrollbar(left, orient="vertical", command=self.rec_tree.yview)
        self.rec_tree.configure(yscrollcommand=sb.set)
        self.rec_tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.rec_tree.bind("<<TreeviewSelect>>", self._on_records_select)

        right = tk.Frame(paned, bg=BG_MAIN)
        paned.add(right, minsize=360)
        self.areas["records_detail"] = ScrollArea(right)
        self._scroll_canvases.add(self.areas["records_detail"].canvas)

        self._records_all = []          # 全部记录(最新在前)
        self._rec_selected_expect = None
        self._records_refresh(first=True)

    def _records_refresh(self, first=False, reload=False):
        """重载开奖记录列表(最新在前), 按期号筛选; 保留选中并重渲染详情。"""
        if first or reload or not self._records_all:
            self._records_all = list(reversed(load_cache()))
        kw = self.rec_filter_var.get().strip()
        rows = [r for r in self._records_all if kw in r.expect] if kw else list(self._records_all)
        prev_sel = self.rec_tree.selection()
        prev_id = prev_sel[0] if prev_sel else self._rec_selected_expect
        for iid in self.rec_tree.get_children():
            self.rec_tree.delete(iid)
        for r in rows:
            self.rec_tree.insert("", "end", iid=r.expect, values=(
                r.expect, (r.open_time or "")[:16],
                ",".join(f"{n:02d}" for n in r.regular), f"{r.special:02d}"))
        total = len(self._records_all)
        self.rec_status.config(text=f"共 {len(rows)} 期" + (f" (筛选自 {total})" if kw else ""))
        want = None
        if prev_id and self.rec_tree.exists(prev_id):
            want = prev_id
        elif first and rows:
            want = rows[0].expect
        if want:
            self._rec_selected_expect = want
            self.rec_tree.selection_set(want)
            self.rec_tree.see(want)
            rec = next((r for r in rows if r.expect == want), None)
            if rec:
                self._render_record_detail(rec)
        elif not rows:
            a = self.areas["records_detail"]
            a.clear()
            tk.Label(a.inner, text="无匹配期号。", bg=BG_MAIN, fg=MUTED,
                     font=FONT).pack(anchor="w", padx=18, pady=24)

    def _on_records_select(self, _event=None):
        sel = self.rec_tree.selection()
        if not sel:
            return
        self._rec_selected_expect = sel[0]
        rec = next((r for r in self._records_all if r.expect == sel[0]), None)
        if rec:
            self._render_record_detail(rec)

    def _render_record_detail(self, rec):
        """只读渲染一期开奖: 波色球 + 生肖/波色 + 特码统计。"""
        a = self.areas["records_detail"]
        a.clear()
        head = self._card(a.inner, f"第 {rec.expect} 期开奖")
        tk.Label(head, text=f"开奖时间: {rec.open_time}", bg=BG_CARD, fg=MUTED,
                 font=FONT).pack(anchor="w", pady=(2, 4))
        # 开奖号码(球) + 生肖 + 波色: grid 三行共享列, 生肖/波色居中其下, 与号码球上下对齐
        zs = rec.zodiacs if len(rec.zodiacs) >= 7 else (rec.zodiacs + [""] * 7)
        grid = tk.Frame(head, bg=BG_CARD)
        grid.pack(fill="x", pady=(2, 6))
        ZOD_FONT = ("Microsoft YaHei UI", 20, "bold")
        for ri, lab in enumerate(("开奖", "生肖", "波色")):
            tk.Label(grid, text=lab, bg=BG_CARD, fg=MUTED, font=FONT_SM,
                     width=4).grid(row=ri, column=0, sticky="w", padx=(0, 4))
        # 6 平码列
        for ci, n in enumerate(rec.regular, start=1):
            make_ball(grid, f"{n:02d}", "reg_l").grid(row=0, column=ci, padx=3, pady=2)
            tk.Label(grid, text=zs[ci - 1] or "—", bg=BG_CARD, fg=PRIMARY_L,
                     font=ZOD_FONT).grid(row=1, column=ci)
            tk.Label(grid, text=wave_of_number(n) or "—", bg=BG_CARD, fg=MUTED,
                     font=FONT_SM).grid(row=2, column=ci)
        # "+" 分隔(仅号码行)
        tk.Label(grid, text="+", bg=BG_CARD, fg=ACCENT, font=FONT_H).grid(row=0, column=7, padx=2)
        # 特码列
        make_ball(grid, f"{rec.special:02d}", "special_l").grid(row=0, column=8, padx=3, pady=2)
        tk.Label(grid, text=zs[6] or "—", bg=BG_CARD, fg=PRIMARY_L,
                 font=ZOD_FONT).grid(row=1, column=8)
        tk.Label(grid, text=wave_of_number(rec.special) or "—", bg=BG_CARD, fg=MUTED,
                 font=FONT_SM).grid(row=2, column=8)
        # 特码统计
        sp = rec.special
        self._metric_row(head, [
            ("平码和值", f"{sum(rec.regular)}", TEXT),
            ("特码大小", "大" if sp >= 25 else "小", TEXT),
            ("特码奇偶", "奇" if sp % 2 == 1 else "偶", TEXT),
            ("特码尾数", f"{sp % 10}", TEXT),
            ("特码头数", f"{sp // 10}", TEXT),
        ])
        dcard = self._card(a.inner, "科学提示", accent=MUTED)
        tk.Label(dcard, text=DISCLAIMER, bg=BG_CARD, fg=MUTED, font=FONT_SM,
                 justify="left", anchor="w").pack(fill="x")

    # ---------------- 历史预测页 ----------------
    def _build_history_page(self, page):
        """历史预测页: 左侧列表(每期一行) + 右侧详情(只读渲染 + 命中核对)。"""
        bar = tk.Frame(page, bg=BG_MAIN)
        bar.pack(fill="x", padx=12, pady=(8, 4))
        b1 = tk.Button(bar, text="⟳ 刷新核对", command=self._history_fetch_fresh,
                       bg=ACCENT, fg="#ffffff", activebackground=PRIMARY,
                       activeforeground="#ffffff", relief="flat", bd=0,
                       font=("Microsoft YaHei UI", 9, "bold"), padx=12, pady=4, cursor="hand2")
        b1.pack(side="left")
        self._hover(b1, ACCENT, PRIMARY)
        b2 = tk.Button(bar, text="🗑 删除选中", command=self._history_delete,
                       bg=BG_CARD2, fg=TEXT, activebackground=PRIMARY_L,
                       relief="flat", bd=0, font=FONT_SM, padx=12, pady=4, cursor="hand2")
        b2.pack(side="left", padx=6)
        b3 = tk.Button(bar, text="清空历史", command=self._history_clear,
                       bg=BG_CARD2, fg=WARN, activebackground=PRIMARY_L,
                       relief="flat", bd=0, font=FONT_SM, padx=12, pady=4, cursor="hand2")
        b3.pack(side="left", padx=6)
        self.hist_status = tk.Label(bar, text="", bg=BG_MAIN, fg=MUTED, font=FONT_SM)
        self.hist_status.pack(side="left", padx=14)

        paned = tk.PanedWindow(page, orient="horizontal", bg=BG_MAIN, sashwidth=6,
                               sashrelief="flat")
        paned.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        left = tk.Frame(paned, bg=BG_MAIN)
        # paned.add(left, minsize=300, width=360)
        paned.add(left, minsize=300, width=400)
        cols = ("time", "target", "special", "wide", "zodiac")
        headers = {"time": "时间", "target": "目标/状态",
                   "special": "推荐特码", "wide": "大集合", "zodiac": "生肖"}
        self.hist_tree = ttk.Treeview(left, columns=cols, show="headings", height=22)
        for c in cols:
            self.hist_tree.heading(c, text=headers[c])
        self.hist_tree.column("time", width=120, anchor="w")
        self.hist_tree.column("target", width=110, anchor="w")
        self.hist_tree.column("special", width=60, anchor="center")
        self.hist_tree.column("wide", width=55, anchor="center")
        self.hist_tree.column("zodiac", width=50, anchor="center")
        self.hist_tree.tag_configure("pending", foreground=TEXT2)
        self.hist_tree.tag_configure("hit", foreground=OK)
        sb = ttk.Scrollbar(left, orient="vertical", command=self.hist_tree.yview)
        self.hist_tree.configure(yscrollcommand=sb.set)
        self.hist_tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.hist_tree.bind("<<TreeviewSelect>>", self._on_history_select)

        right = tk.Frame(paned, bg=BG_MAIN)
        paned.add(right, minsize=380)
        self.areas["history_detail"] = ScrollArea(right)
        self._scroll_canvases.add(self.areas["history_detail"].canvas)

        self._hist_runs = []      # 列表对应的 run 字典(最新在前)
        self._hist_verify = {}    # run_id -> verify 结果(或 None=待开奖)
        self._hist_records = None  # 核对用开奖记录(刷新核对时更新为最新); None 时用缓存
        self._hist_selected_id = None
        self._hist_fetching = False
        self._history_refresh(first=True)

    def _on_tab_changed(self, _event=None):
        """切到历史预测页/开奖记录页时刷新(开奖记录重读缓存以反映录入/抓取)。"""
        sel = self.nb.select()
        if sel == str(self._hist_page):
            self._history_refresh()
        elif sel == str(self._records_page):
            self._records_refresh(reload=True)

    def _history_refresh(self, first=False):
        """重载历史列表并核对每期命中。

        核对用 self._hist_records(刷新核对时拉取的最新开奖); 未拉取时回退本地缓存。
        刷新后保留之前选中的记录并重渲染其详情(命中状态随之更新)。
        """
        records = self._hist_records if self._hist_records is not None else load_cache()
        runs = load_runs()
        self._hist_runs = list(reversed(runs))  # 最新在前
        self._hist_verify = {r.get("run_id", ""): verify_run(r, records)
                             for r in self._hist_runs}
        # 记住当前选中, 以便刷新后恢复
        prev_sel = self.hist_tree.selection()
        prev_id = prev_sel[0] if prev_sel else self._hist_selected_id
        for iid in self.hist_tree.get_children():
            self.hist_tree.delete(iid)
        for r in self._hist_runs:
            rid = r.get("run_id", "")
            v = self._hist_verify.get(rid)
            g0 = r.get("groups", [{}])[0] if r.get("groups") else {}
            if v is None:
                # 未开奖: 目标期 = 基准期(basis_expect)的下一期; basis 为空/非数字时回退无期号文案
                basis = r.get("basis_expect", "")
                try:
                    next_expect = str(int(basis) + 1) if basis else ""
                except ValueError:
                    next_expect = ""
                target = f"{next_expect} 待开奖" if next_expect else "待开奖"
                wide_t, zod_t, tag = "—", "—", "pending"
            else:
                target = f"{v['actual_expect']} 已开"
                wide_t = "✓" if v["wide_hit"] else "✗"
                zod_t = "✓" if v["zodiac_hit"] else "✗"
                tag = "hit" if (v["wide_hit"] or v["zodiac_hit"] or v["zodiac_quad_hit"] or v["zodiac_six_hit"]
                                or v["special_hit_any"]) else ""
            self.hist_tree.insert("", "end", iid=rid, tags=(tag,) if tag else (),
                                  values=(r.get("created_at", "")[:16],
                                          target,
                                          f"{g0.get('special', 0):02d}", wide_t, zod_t))
        n = len(self._hist_runs)
        n_open = sum(1 for vv in self._hist_verify.values() if vv is None)
        latest_txt = ""
        if records:
            latest_txt = f" · 开奖至 {records[-1].expect}"
        self.hist_status.config(text=f"共 {n} 期 · 已开奖 {n - n_open} · 待开奖 {n_open}{latest_txt}")
        # 恢复选中并重渲染详情
        want_id = None
        if prev_id and any(r.get("run_id") == prev_id for r in self._hist_runs):
            want_id = prev_id
        elif first and self._hist_runs:
            want_id = self._hist_runs[0].get("run_id", "")
        if want_id:
            self._hist_selected_id = want_id
            self.hist_tree.selection_set(want_id)
            run = next((r for r in self._hist_runs if r.get("run_id") == want_id), None)
            if run:
                self._render_history_detail(run)
        elif not self._hist_runs:
            a = self.areas["history_detail"]
            a.clear()
            tk.Label(a.inner, text="暂无历史预测。\n\n点击「▶ 开始分析」运行一次预测, "
                     "结果会自动保存于此, 供日后查阅与核对。",
                     bg=BG_MAIN, fg=MUTED, font=FONT, justify="left").pack(
                         anchor="w", padx=18, pady=24)

    def _history_fetch_fresh(self):
        """后台拉取最新开奖并重新核对(联网, 不卡 UI)。"""
        if self._hist_fetching:
            return
        self._hist_fetching = True
        self.hist_status.config(text="正在获取最新开奖...")
        self._status("获取最新开奖中...", ACCENT)

        def work():
            try:
                recs = load_history(HISTORY_YEARS, refresh=False)
                self.queue.put(("hist_verified", recs))
            except Exception as e:
                self.queue.put(("hist_verified", None))
                print(f"  ! 历史页刷新核对失败: {e}")

        threading.Thread(target=work, daemon=True).start()

    def _on_history_select(self, _event=None):
        sel = self.hist_tree.selection()
        if not sel:
            return
        self._hist_selected_id = sel[0]
        run = next((r for r in self._hist_runs if r.get("run_id") == sel[0]), None)
        if run:
            self._render_history_detail(run)

    def _history_delete(self):
        sel = self.hist_tree.selection()
        if not sel:
            messagebox.showinfo("提示", "请先在列表中选择一条记录。")
            return
        if not messagebox.askyesno("确认", "删除选中的这条预测记录?"):
            return
        delete_run(sel[0])
        self._history_refresh()

    def _history_clear(self):
        if not load_runs():
            return
        if not messagebox.askyesno("确认", "清空全部历史预测记录?\n此操作不可撤销。"):
            return
        clear_runs()
        self._history_refresh()

    def _render_history_detail(self, run):
        """只读渲染一期预测的全部明细, 并标注命中情况。"""
        a = self.areas["history_detail"]
        a.clear()
        rid = run.get("run_id", "")
        v = self._hist_verify.get(rid)

        # 预测目标期号: 已开奖取实际期号, 否则取基准期下一期(basis 为空/非数字时留空)
        if v is not None:
            target_expect = v["actual_expect"]
        else:
            basis = run.get("basis_expect", "")
            try:
                target_expect = str(int(basis) + 1) if basis else ""
            except ValueError:
                target_expect = ""

        # 概览
        head = self._card(a.inner, "预测概览", expect=target_expect)
        self._metric_row(head, [
            ("生成时间", run.get("created_at", "")[:16], TEXT),
            ("基于期号", run.get("basis_expect", ""), TEXT),
            ("历史期数", f"{run.get('history_count', 0)}", TEXT),
            ("回测期数", f"{run.get('backtest_n', 0)}", TEXT),
        ])
        if v is None:
            target_txt = "待开奖(目标期尚未开出)"
            target_col = TEXT2
        else:
            target_txt = f"{v['actual_expect']}  实际特码 {v['actual_special']:02d}"
            target_col = ACCENT
        tk.Label(head, text="预测目标: " + target_txt, bg=BG_CARD, fg=target_col,
                 font=FONT).pack(anchor="w", pady=(6, 0))
        if run.get("llm_models"):
            tk.Label(head, text="反推模型: " + ", ".join(run["llm_models"]), bg=BG_CARD,
                     fg=PRIMARY_L, font=FONT_SM, wraplength=820, justify="left",
                     anchor="w").pack(fill="x", pady=(4, 0))

        # 命中概览
        if v is not None:
            hcard = self._card(a.inner, "命中核对", accent=ACCENT, expect=target_expect)
            self._metric_row(hcard, [
                ("特码", "✓" if v["full_special_hit"] else "✗",
                 OK if v["full_special_hit"] else WARN),
                ("平码", f"{v['full_regular_hits']}/6", TEXT),
                ("五组特码", "✓" if v["special_hit_any"] else "✗",
                 OK if v["special_hit_any"] else WARN),
                ("大集合", "✓" if v["wide_hit"] else "✗",
                 OK if v["wide_hit"] else WARN),
                ("三生肖", "✓" if v["zodiac_hit"] else "✗",
                 OK if v["zodiac_hit"] else WARN),
                ("四生肖", "✓" if v["zodiac_quad_hit"] else "✗",
                 OK if v["zodiac_quad_hit"] else WARN),
                ("六生肖", "✓" if v["zodiac_six_hit"] else "✗",
                 OK if v["zodiac_six_hit"] else WARN),
                ("六维度", f"{v['dim_hit_count']}/{v['dim_total']}", TEXT),
            ])

        actual_reg = set(v["actual_regular"]) if v else set()
        actual_sp = v["actual_special"] if v else None

        # 三组完整预测
        for i, g in enumerate(run.get("groups", [])):
            body = self._card(a.inner, f"{g.get('name', '')}组  完整预测", star=(i == 0), expect=target_expect)
            brow = tk.Frame(body, bg=BG_CARD)
            brow.pack(fill="x", pady=(2, 4))
            tk.Label(brow, text="平码", bg=BG_CARD, fg=MUTED, font=FONT_SM, width=4).pack(side="left")
            for n in g.get("regular", []):
                add_ball(brow, f"{n:02d}", "reg", hit=bool(v and n in actual_reg))
            srow = tk.Frame(body, bg=BG_CARD)
            srow.pack(fill="x", pady=(4, 6))
            tk.Label(srow, text="特码", bg=BG_CARD, fg=MUTED, font=FONT_SM, width=4).pack(side="left")
            sp_hit = v and g.get("special") == actual_sp
            add_ball(srow, f"{g.get('special', 0):02d}", "special", hit=bool(sp_hit))
            tk.Label(srow, text=f"  {g.get('strategy', '')}", bg=BG_CARD, fg=MUTED,
                     font=FONT_SM).pack(side="left", padx=8)
            self._metric_row(body, [
                ("特码命中率", f"{g.get('backtest_special_hit', 0):.1%}", PRIMARY_L),
                ("平均平码命中", f"{g.get('backtest_regular_hits', 0):.2f}/6", TEXT),
            ])

        # 五组单颗特码
        for g in run.get("specials", []):
            body = self._card(a.inner, f"特码 {g.get('name', '')}组", expect=target_expect)
            row = tk.Frame(body, bg=BG_CARD)
            row.pack(fill="x", pady=(2, 6))
            hit = v and g.get("special") == actual_sp
            add_ball(row, f"{g.get('special', 0):02d}", "special", hit=bool(hit))
            tk.Label(row, text=f"  {g.get('strategy', '')}" + ("  ✓ 命中" if hit else ""),
                     bg=BG_CARD, fg=(OK if hit else MUTED), font=FONT_SM).pack(side="left", padx=6)
            self._metric_row(body, [
                ("命中率", f"{g.get('backtest_hit', 0):.1%}", PRIMARY_L),
                ("提升度", f"{g.get('lift', 0):+.1%}", self._lift_color(g.get('lift', 0))),
            ])

        # 六维度
        dims = run.get("dims", [])
        if dims:
            body = self._card(a.inner, "六维度特码属性", expect=target_expect)
            hrow = tk.Frame(body, bg=BG_CARD)
            hrow.pack(fill="x", pady=(2, 4))
            for h, w in [("维度", 8), ("预测值", 8), ("实际", 8),
                         ("命中", 8), ("准确率", 10), ("提升度", 10)]:
                tk.Label(hrow, text=h, bg=BG_CARD, fg=MUTED, font=FONT_SM,
                         width=w, anchor="w").pack(side="left")
            for d in dims:
                name = d.get("name", "")
                hit = bool(v and v.get("dim_hits", {}).get(name))
                actual_dim = v["dim_actuals"].get(name, "—") if v else "—"
                row = tk.Frame(body, bg=BG_CARD)
                row.pack(fill="x", pady=2)
                vals = [name, str(d.get("value", "")), actual_dim,
                        "✓" if hit else ("✗" if v else "—"),
                        f"{d.get('accuracy', 0):.1%}", f"{d.get('lift', 0):+.1%}"]
                colors = [TEXT, PRIMARY_L, TEXT,
                          (OK if hit else WARN), PRIMARY_L, self._lift_color(d.get('lift', 0))]
                widths = [8, 8, 8, 8, 10, 10]
                for val, c, w in zip(vals, colors, widths):
                    tk.Label(row, text=val, bg=BG_CARD, fg=c, font=FONT,
                             width=w, anchor="w").pack(side="left")

        # 号码集合
        for i, p in enumerate(run.get("pools", [])):
            nums = p.get("numbers", [])
            body = self._card(a.inner, f"集合 {p.get('name', '')}组  ({len(nums)}颗)",
                              star=(i == 0), expect=target_expect)
            row = tk.Frame(body, bg=BG_CARD)
            row.pack(fill="x", pady=(2, 6))
            p_hit = v and actual_sp in set(nums)
            for n in nums:
                add_ball(row, f"{n:02d}", "pool", hit=bool(v and n == actual_sp))
            if v:
                tk.Label(row, text=f"  {'✓ 含特码' if p_hit else '✗ 未含'}", bg=BG_CARD,
                         fg=(OK if p_hit else WARN), font=FONT_SM).pack(side="left", padx=6)
            self._metric_row(body, [
                ("命中率", f"{p.get('hit_rate', 0):.1%}", PRIMARY_L),
                ("提升度", f"{p.get('lift', 0):+.1%}", self._lift_color(p.get('lift', 0))),
            ])

        # 20 颗大集合
        wide = run.get("wide", {})
        if wide:
            nums = wide.get("numbers", [])
            body = self._card(a.inner, "20 颗特码大集合", star=True, accent=ACCENT, expect=target_expect)
            row = tk.Frame(body, bg=BG_CARD)
            row.pack(fill="x", pady=(2, 6))
            w_hit = v and actual_sp in set(nums)
            for n in nums:
                add_ball(row, f"{n:02d}", "wide", hit=bool(v and n == actual_sp))
            if v:
                tk.Label(row, text=f"  {'✓ 含特码' if w_hit else '✗ 未含'}", bg=BG_CARD,
                         fg=(OK if w_hit else WARN), font=FONT_SM).pack(side="left", padx=6)
            self._metric_row(body, [
                ("命中率", f"{wide.get('hit_rate', 0):.1%}", PRIMARY_L),
                ("提升度", f"{wide.get('lift', 0):+.1%}", self._lift_color(wide.get('lift', 0))),
            ])

        # 三生肖
        zod = run.get("zodiac", {})
        if zod:
            body = self._card(a.inner, "特码三生肖", star=True, accent=ACCENT, expect=target_expect)
            row = tk.Frame(body, bg=BG_CARD)
            row.pack(fill="x", pady=(2, 8))
            z_hit = v and v.get("actual_zodiac") in set(zod.get("zodiacs", []))
            for z in zod.get("zodiacs", []):
                add_ball(row, z, "zodiac")
            if v:
                tk.Label(row, text=f"  实际生肖: {v.get('actual_zodiac', '')}  "
                         f"{'✓ 命中' if z_hit else '✗ 未中'}", bg=BG_CARD,
                         fg=(OK if z_hit else WARN), font=FONT_SM).pack(side="left", padx=6)
            self._metric_row(body, [
                ("命中率", f"{zod.get('hit_rate', 0):.1%}", PRIMARY_L),
                ("提升度", f"{zod.get('lift', 0):+.1%}", self._lift_color(zod.get('lift', 0))),
            ])

        # 四生肖
        zodq = run.get("zodiac_quad", {})
        if zodq:
            body = self._card(a.inner, "特码四生肖", star=True, accent=ACCENT, expect=target_expect)
            row = tk.Frame(body, bg=BG_CARD)
            row.pack(fill="x", pady=(2, 8))
            zq_hit = v and v.get("actual_zodiac") in set(zodq.get("zodiacs", []))
            for z in zodq.get("zodiacs", []):
                add_ball(row, z, "zodiac")
            if v:
                tk.Label(row, text=f"  实际生肖: {v.get('actual_zodiac', '')}  "
                         f"{'✓ 命中' if zq_hit else '✗ 未中'}", bg=BG_CARD,
                         fg=(OK if zq_hit else WARN), font=FONT_SM).pack(side="left", padx=6)
            self._metric_row(body, [
                ("命中率", f"{zodq.get('hit_rate', 0):.1%}", PRIMARY_L),
                ("提升度", f"{zodq.get('lift', 0):+.1%}", self._lift_color(zodq.get('lift', 0))),
            ])

        # 六生肖
        zods = run.get("zodiac_six", {})
        if zods:
            body = self._card(a.inner, "特码六生肖", star=True, accent=ACCENT, expect=target_expect)
            row = tk.Frame(body, bg=BG_CARD)
            row.pack(fill="x", pady=(2, 8))
            zs_hit = v and v.get("actual_zodiac") in set(zods.get("zodiacs", []))
            for z in zods.get("zodiacs", []):
                add_ball(row, z, "zodiac")
            if v:
                tk.Label(row, text=f"  实际生肖: {v.get('actual_zodiac', '')}  "
                         f"{'✓ 命中' if zs_hit else '✗ 未中'}", bg=BG_CARD,
                         fg=(OK if zs_hit else WARN), font=FONT_SM).pack(side="left", padx=6)
            self._metric_row(body, [
                ("命中率", f"{zods.get('hit_rate', 0):.1%}", PRIMARY_L),
                ("提升度", f"{zods.get('lift', 0):+.1%}", self._lift_color(zods.get('lift', 0))),
            ])

        dcard = self._card(a.inner, "科学提示", accent=MUTED)
        tk.Label(dcard, text=DISCLAIMER, bg=BG_CARD, fg=MUTED, font=FONT_SM,
                 justify="left", anchor="w").pack(fill="x")

    # ---------------- 手工录入开奖 ----------------
    def _manual_entry(self):
        """打开手工录入当期开奖对话框: 期号 + 7 个号码, 波色自动着色, 生肖取自近期开奖。"""
        win = tk.Toplevel(self.root)
        win.title("手工录入开奖记录")
        win.geometry("540x440")
        win.configure(bg=BG_MAIN)
        win.transient(self.root)
        win.grab_set()

        recs = load_cache()
        zmap = build_zodiac_map(recs)
        # 期号预填: 上一期 + 1
        sug = ""
        if recs:
            try:
                sug = str(int(recs[-1].expect) + 1)
            except ValueError:
                sug = recs[-1].expect

        form = tk.Frame(win, bg=BG_MAIN)
        form.pack(fill="x", padx=18, pady=(16, 4))
        tk.Label(form, text="期号", bg=BG_MAIN, fg=TEXT, font=FONT,
                 width=10, anchor="w").grid(row=0, column=0, sticky="w")
        exp_var = tk.StringVar(value=sug)
        tk.Entry(form, textvariable=exp_var, width=16, font=FONT, relief="solid",
                 borderwidth=1, bg=BG_CARD, fg=TEXT, insertbackground=TEXT).grid(
                     row=0, column=1, sticky="w", padx=4)
        tk.Label(form, text="开奖号码", bg=BG_MAIN, fg=TEXT, font=FONT,
                 width=10, anchor="w").grid(row=1, column=0, sticky="nw", pady=(10, 0))
        num_var = tk.StringVar()
        tk.Entry(form, textvariable=num_var, width=30, font=("Consolas", 11),
                 relief="solid", borderwidth=1, bg=BG_CARD, fg=TEXT,
                 insertbackground=TEXT).grid(row=1, column=1, sticky="w", padx=4, pady=(10, 0))
        tk.Label(form, text="6平码+1特码, 逗号分隔, 如 44,24,11,36,25,20,01",
                 bg=BG_MAIN, fg=MUTED, font=FONT_SM).grid(row=2, column=1, sticky="w", padx=4)
        tk.Label(form, text="开奖时间", bg=BG_MAIN, fg=TEXT, font=FONT,
                 width=10, anchor="w").grid(row=3, column=0, sticky="w", pady=(10, 0))
        time_var = tk.StringVar(value=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        tk.Entry(form, textvariable=time_var, width=22, font=FONT, relief="solid",
                 borderwidth=1, bg=BG_CARD, fg=TEXT, insertbackground=TEXT).grid(
                     row=3, column=1, sticky="w", padx=4, pady=(10, 0))

        pv = self._card(win, "解析预览  (波色按号码自动着色 · 生肖取自近期开奖)")
        preview = tk.Frame(pv, bg=BG_CARD)
        preview.pack(fill="x", pady=4)

        def render_preview(*_):
            for w in preview.winfo_children():
                w.destroy()
            raw = num_var.get().replace("，", ",").replace(" ", ",")
            try:
                nums = [int(x) for x in raw.split(",") if x.strip() != ""]
            except ValueError:
                nums = []
            if len(nums) != 7 or any(not (1 <= n <= 49) for n in nums):
                tk.Label(preview, text="请输入 7 个 1-49 的号码(逗号分隔)",
                         bg=BG_CARD, fg=MUTED, font=FONT_SM).pack(anchor="w")
                return
            row = tk.Frame(preview, bg=BG_CARD)
            row.pack(fill="x", pady=2)
            for i, n in enumerate(nums):
                add_ball(row, f"{n:02d}", "special" if i == 6 else "reg")
            tk.Label(preview, text="生肖: " + "  ".join(zmap.get(n, "—") for n in nums),
                     bg=BG_CARD, fg=PRIMARY_L, font=FONT_SM).pack(anchor="w", pady=(4, 0))
            tk.Label(preview, text="波色: " + "  ".join(wave_of_number(n) or "—" for n in nums),
                     bg=BG_CARD, fg=MUTED, font=FONT_SM).pack(anchor="w")

        num_var.trace_add("write", render_preview)
        render_preview()

        btns = tk.Frame(win, bg=BG_MAIN)
        btns.pack(fill="x", padx=18, pady=(6, 16))
        tk.Button(btns, text="取消", bg=BG_CARD2, fg=TEXT, relief="flat", bd=0,
                  font=FONT, padx=16, pady=6, cursor="hand2",
                  command=win.destroy).pack(side="right", padx=6)
        ok = tk.Button(btns, text="保存并核对", bg=ACCENT, fg="#ffffff", relief="flat", bd=0,
                       font=("Microsoft YaHei UI", 10, "bold"), padx=18, pady=6, cursor="hand2",
                       command=lambda: self._submit_entry(win, exp_var, num_var, time_var, zmap))
        ok.pack(side="right")
        self._hover(ok, ACCENT, PRIMARY)
        win.bind("<Escape>", lambda e: win.destroy())

    def _submit_entry(self, win, exp_var, num_var, time_var, zmap):
        """校验并保存录入的开奖记录到缓存, 随后刷新历史核对。"""
        expect = exp_var.get().strip()
        if not expect:
            messagebox.showerror("期号缺失", "请填写期号。", parent=win)
            return
        raw = num_var.get().replace("，", ",").replace(" ", ",")
        try:
            nums = [int(x) for x in raw.split(",") if x.strip() != ""]
        except ValueError:
            messagebox.showerror("号码格式错误", "请输入数字, 逗号分隔。", parent=win)
            return
        if len(nums) != 7:
            messagebox.showerror("号码数量错误", "需输入 7 个号码(6平码+1特码)。", parent=win)
            return
        if any(not (1 <= n <= 49) for n in nums):
            messagebox.showerror("号码范围错误", "号码需在 1-49 之间。", parent=win)
            return
        if len(set(nums)) != 7:
            if not messagebox.askyesno("号码重复", "存在重复号码, 仍要保存吗?", parent=win):
                return
        cached = load_cache()
        if any(r.expect == expect for r in cached):
            if not messagebox.askyesno("期号已存在",
                                       f"期号 {expect} 已有记录, 是否覆盖?", parent=win):
                return
        regular, special = nums[:6], nums[6]
        rec = Record(expect=expect,
                     open_time=time_var.get().strip() or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                     regular=regular, special=special,
                     waves=[wave_of_number(n) for n in nums],
                     zodiacs=[zmap.get(n, "") for n in nums])
        try:
            save_cache(merge_records([cached, [rec]]))
        except Exception as e:
            messagebox.showerror("保存失败", f"{type(e).__name__}: {e}", parent=win)
            return
        if self._hist_records is not None:
            self._hist_records = merge_records([self._hist_records, [rec]])
        self._set_next_issue(load_cache())
        win.destroy()
        self.nb.select(self._hist_page)
        self._history_refresh()
        self._status(f"已录入 {expect} 期开奖记录", OK)


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
