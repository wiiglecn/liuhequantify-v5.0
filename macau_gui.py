#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""澳门六合彩分析与预测 - 图形界面版(tkinter)
白底清爽主题 + 淡红主色调 + 3D 立体号码球 + 卡片流排版。"""
import os
import sys
import threading
import queue

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        os.system("chcp 65001 >nul 2>&1")

import tkinter as tk
from tkinter import ttk, messagebox

from data_fetcher import load_history
from analysis import build_report, summarize_for_llm
from llm_reasoner import load_config, reason
from predictor import predict_all, predict_special_groups
from dimensions import predict_dimensions, predict_zodiac_pool
from special_pool import predict_special_pools, predict_wide_pool

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
    "(特码 1/49≈2%, 单平码 6/49)。命中率基于历史回测, 不代表未来, 仅供研究, 不构成投注建议。"
)


def _hex(s):
    return int(s, 16)


def _blend(c1, c2, t):
    r1, g1, b1 = _hex(c1[1:3]), _hex(c1[3:5]), _hex(c1[5:7])
    r2, g2, b2 = _hex(c2[1:3]), _hex(c2[3:5]), _hex(c2[5:7])
    r = int(r1 + (r2 - r1) * t)
    g = int(g1 + (g2 - g1) * t)
    b = int(b1 + (b2 - b1) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


def draw_sphere(cv, cx, cy, r, text, base, light, tcolor="#ffffff", font=FONT_NUM):
    """在 Canvas 上画一颗 3D 立体球(投影 + 径向渐变 + 高光 + 文字)。"""
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
    gx, gy = cx - r * 0.35, cy - r * 0.45
    cv.create_oval(gx - r * 0.3, gy - r * 0.22, gx + r * 0.3, gy + r * 0.22,
                   fill="#ffffff", outline="", stipple="gray50")
    cv.create_text(cx, cy + 1, text=text, fill=tcolor, font=font)


# 球样式: (base, light, r, font)
BALL_STYLES = {
    "reg": ("#a83a3a", "#e57373", 15, FONT_NUM_S),
    "special": ("#c0392b", "#ff7878", 17, FONT_NUM),
    "pool": ("#9c3535", "#dd6868", 14, FONT_NUM_S),
    "wide": ("#8f3030", "#d06060", 13, FONT_NUM_S),
    "zodiac": ("#a83a3a", "#e57373", 18, FONT_ZOD),
    "muted": ("#3a3a48", "#5a5a6a", 15, FONT_NUM_S),
}


def add_ball(parent, text, kind="reg"):
    base, light, r, font = BALL_STYLES[kind]
    size = r * 2 + 6
    cv = tk.Canvas(parent, width=size, height=size, bg=parent["bg"], highlightthickness=0)
    draw_sphere(cv, size / 2, size / 2, r, text, base, light, "#ffffff", font)
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
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind_all("<MouseWheel>", self._on_wheel)

    def _on_resize(self, e):
        self.canvas.itemconfigure("inner", width=e.width)

    def _on_wheel(self, e):
        self.canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")

    def clear(self):
        for w in self.inner.winfo_children():
            w.destroy()


class App:
    def __init__(self, root):
        self.root = root
        self.queue = queue.Queue()
        self.results = None
        self.worker = None
        root.title("澳门六合彩开奖记录分析与预测")
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

    # ---------------- UI 构建 ----------------
    def _build_ui(self):
        # 顶部标题区
        header = tk.Frame(self.root, bg=BG_MAIN)
        header.pack(fill="x", padx=20, pady=(16, 2))
        tk.Label(header, text="澳门六合彩", fg=ACCENT, bg=BG_MAIN,
                 font=FONT_TITLE).pack(side="left")
        tk.Label(header, text="  开奖记录分析与预测", fg=TEXT, bg=BG_MAIN,
                 font=("Microsoft YaHei UI", 13)).pack(side="left", padx=2)
        tk.Label(header, text="统计模型 + 大模型反推理", fg=MUTED, bg=BG_MAIN,
                 font=FONT_SM).pack(side="left", padx=(10, 0), pady=(4, 0))
        accent = tk.Frame(self.root, bg=PRIMARY_D, height=2)
        accent.pack(fill="x", padx=20, pady=(0, 6))
        # 渐变细线
        glow = tk.Frame(self.root, bg=BORDER, height=1)
        glow.pack(fill="x", padx=20, pady=(0, 6))

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

        tk.Label(bar, text="  回测期数", bg=BG_MAIN, fg=MUTED, font=FONT).pack(side="left", padx=(14, 4))
        self.bt_var = tk.IntVar(value=30)
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
                  ("pools", "号码集合"), ("zodiac", "三生肖")]
        for key, title in titles:
            page = tk.Frame(self.nb, bg=BG_MAIN)
            self.nb.add(page, text=title)
            self.areas[key] = ScrollArea(page)
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

    # ---------------- 卡片构件 ----------------
    def _card(self, parent, title="", star=False, accent=PRIMARY):
        """带投影 + 顶部彩条的立体卡片, 返回内容 Frame。"""
        shadow = tk.Frame(parent, bg=SHADOW)
        shadow.pack(fill="x", padx=14, pady=(8, 12))
        card = tk.Frame(shadow, bg=BG_CARD)
        card.pack(fill="x", padx=(0, 4), pady=(0, 5))
        tk.Frame(card, bg=accent, height=3).pack(fill="x")
        if title:
            hd = tk.Frame(card, bg=BG_CARD)
            hd.pack(fill="x", padx=16, pady=(10, 4))
            txt = title + ("  ★ 推荐" if star else "")
            tk.Label(hd, text=title, bg=BG_CARD, fg=accent, font=FONT_H).pack(side="left")
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
            records = load_history([2024, 2025, 2026], refresh=self.refresh_var.get())
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
            self.queue.put(("done", dict(records=records, report=report, llm=llm_result,
                                         groups=groups, sp=sp, dims=dims,
                                         pools=pools, wide=wide, zp=zp, bt=bt)))
        except Exception as e:
            self.queue.put(("error", f"{type(e).__name__}: {e}"))

    def _msg(self, text):
        self.queue.put(("msg", text))

    def _status(self, text, color=MUTED):
        self.status.config(text="● " + text, fg=color)

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
                    self._render_all()
                    self.prog.stop()
                    self.run_btn.config(state="normal")
                    self._status("完成", OK)
                    self.nb.select(self.areas["overview"].canvas.master)
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
                ("lift", f"{g.lift:+.1%}", self._lift_color(g.lift)),
                ("标准误", f"{g.std_error:.1%}", TEXT),
                ("稳定性", f"{g.stability:.2f}", TEXT),
            ])

    def _render_dims(self, dims):
        a = self.areas["dims"]
        a.clear()
        body = self._card(a.inner, "六维度特码属性  (频率+马尔可夫+趋势 融合)")
        # 表头
        head = tk.Frame(body, bg=BG_CARD)
        head.pack(fill="x", pady=(2, 4))
        for j, h in enumerate(["维度", "预测值", "准确率", "随机基线", "lift", "标准误", "稳定性"]):
            tk.Label(head, text=h, bg=BG_CARD, fg=MUTED, font=FONT_SM,
                     width=10 if j > 1 else 8, anchor="w").pack(side="left")
        for d in dims:
            row = tk.Frame(body, bg=BG_CARD)
            row.pack(fill="x", pady=2)
            vals = [d.name, d.value, f"{d.accuracy:.1%}", f"{d.baseline:.1%}",
                    f"{d.lift:+.1%}", f"{d.std_error:.1%}", f"{d.stability:.2f}"]
            colors = [TEXT, PRIMARY_L, PRIMARY_L, MUTED, self._lift_color(d.lift), TEXT, TEXT]
            widths = [8, 8, 10, 10, 10, 10, 10]
            for v, c, w in zip(vals, colors, widths):
                tk.Label(row, text=v, bg=BG_CARD, fg=c, font=FONT, width=w, anchor="w").pack(side="left")
        tk.Label(body, text="lift > 0 表示优于随机基线", bg=BG_CARD, fg=MUTED, font=FONT_SM).pack(anchor="w", pady=(8, 0))

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
                ("lift", f"{p.lift:+.1%}", self._lift_color(p.lift)),
                ("稳定性", f"{p.stability:.2f}", TEXT),
            ])
        # 20 颗大集合
        body = self._card(a.inner, f"20 颗特码大集合  (单组 · 多信号融合)", star=True, accent=ACCENT)
        row = tk.Frame(body, bg=BG_CARD)
        row.pack(fill="x", pady=(2, 6))
        for n in wide.numbers:
            add_ball(row, f"{n:02d}", "wide")
        self._metric_row(body, [
            ("命中率", f"{wide.hit_rate:.1%}", PRIMARY_L),
            ("基线", f"{wide.baseline:.1%}", MUTED),
            ("lift", f"{wide.lift:+.1%}", self._lift_color(wide.lift)),
            ("稳定性", f"{wide.stability:.2f}", TEXT),
        ])

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
            ("lift", f"{zp.lift:+.1%}", self._lift_color(zp.lift)),
            ("标准误", f"{zp.std_error:.1%}", TEXT),
            ("稳定性", f"{zp.stability:.2f}", TEXT),
        ])
        dcard = self._card(a.inner, "科学提示", accent=MUTED)
        tk.Label(dcard, text=DISCLAIMER, bg=BG_CARD, fg=MUTED, font=FONT_SM,
                 justify="left", anchor="w").pack(fill="x")


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
