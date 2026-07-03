#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""澳门六合彩分析与预测 - 图形界面版(tkinter)
科技感深色主题 + 淡红色主色调, 卡片式排版。"""
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
from tkinter import ttk, scrolledtext, messagebox

from data_fetcher import load_history
from analysis import build_report, summarize_for_llm
from llm_reasoner import load_config, reason
from predictor import predict_all, predict_special_groups
from dimensions import predict_dimensions, predict_zodiac_pool
from special_pool import predict_special_pools, predict_wide_pool

# ======================== 主题配色 ========================
BG_MAIN = "#14141c"        # 主背景(深)
BG_CARD = "#1e1e2a"        # 卡片背景
BG_CARD2 = "#262635"       # 卡片次级
BORDER = "#3a2a33"         # 边框(带红调)
PRIMARY = "#e57373"        # 淡红主色
PRIMARY_D = "#c05050"      # 深红(按钮按下/边线)
PRIMARY_L = "#f3b4b4"      # 浅红(号码/键名)
ACCENT = "#ff6b6b"         # 霓虹红(高亮/推荐)
TEXT = "#ecedf2"           # 主文字
MUTED = "#8a8a9c"          # 次要文字
OK = "#7ee787"             # 正向(lift>0)
WARN = "#f5a623"           # 警示

FONT = ("Microsoft YaHei UI", 10)
FONT_TITLE = ("Microsoft YaHei UI", 17, "bold")
FONT_H = ("Microsoft YaHei UI", 12, "bold")
FONT_MONO = ("Consolas", 11)
FONT_MONO_B = ("Consolas", 11, "bold")

DISCLAIMER = (
    "科学提示: 彩票开奖本质为独立随机事件, 任何模型的真实命中率理论上均接近随机概率\n"
    "(特码 1/49≈2%, 单平码 6/49)。命中率基于历史回测, 不代表未来, 仅供研究, 不构成投注建议。"
)


def _fmt_nums(nums):
    return ",".join(f"{n:02d}" for n in nums)


class App:
    def __init__(self, root):
        self.root = root
        self.queue = queue.Queue()
        self.results = None
        self.worker = None
        root.title("澳门六合彩开奖记录分析与预测")
        root.geometry("1020x720")
        root.configure(bg=BG_MAIN)
        self._setup_style()
        self._build_ui()
        self._poll()

    # ---------------- 样式 ----------------
    def _setup_style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TNotebook", background=BG_MAIN, borderwidth=0, tabmargins=(6, 6, 6, 0))
        style.configure("TNotebook.Tab", background=BG_MAIN, foreground=MUTED,
                        padding=(18, 8), font=FONT, borderwidth=0)
        style.map("TNotebook.Tab",
                  background=[("selected", BG_CARD)],
                  foreground=[("selected", PRIMARY)],
                  expand=[("selected", (1, 1, 1, 0))])
        style.configure("TCheckbutton", background=BG_MAIN, foreground=TEXT, font=FONT)
        style.map("TCheckbutton", background=[("active", BG_MAIN)])
        style.configure("TSpinbox", fieldbackground=BG_CARD2, foreground=TEXT,
                        background=BG_MAIN, arrowcolor=PRIMARY, bordercolor=BORDER,
                        lightcolor=BORDER, darkcolor=BORDER, insertcolor=PRIMARY)
        style.configure("Horizontal.TProgressbar", background=PRIMARY,
                        troughcolor=BG_CARD2, borderwidth=0, lightcolor=PRIMARY,
                        darkcolor=PRIMARY)

    # ---------------- UI 构建 ----------------
    def _build_ui(self):
        # 顶部标题区
        header = tk.Frame(self.root, bg=BG_MAIN)
        header.pack(fill="x", padx=18, pady=(14, 4))
        tk.Label(header, text="澳门六合彩 · 开奖记录分析与预测", fg=PRIMARY,
                 bg=BG_MAIN, font=FONT_TITLE).pack(side="left")
        tk.Label(header, text="  统计模型 + 大模型反推理", fg=MUTED,
                 bg=BG_MAIN, font=FONT).pack(side="left", padx=4)
        accent = tk.Frame(self.root, bg=PRIMARY_D, height=2)
        accent.pack(fill="x", padx=18, pady=(0, 8))

        # 控制栏
        bar = tk.Frame(self.root, bg=BG_MAIN)
        bar.pack(fill="x", padx=18, pady=4)

        self.run_btn = tk.Button(bar, text="▶  开始分析", command=self.start,
                                 bg=PRIMARY, fg="#1a1a22", activebackground=ACCENT,
                                 activeforeground="#1a1a22", relief="flat", bd=0,
                                 font=("Microsoft YaHei UI", 10, "bold"),
                                 padx=16, pady=5, cursor="hand2")
        self.run_btn.pack(side="left")
        self._add_hover(self.run_btn, PRIMARY, ACCENT)

        tk.Label(bar, text="  回测期数", bg=BG_MAIN, fg=MUTED, font=FONT).pack(side="left", padx=(14, 4))
        self.bt_var = tk.IntVar(value=30)
        ttk.Spinbox(bar, from_=5, to=100, textvariable=self.bt_var, width=5,
                    font=FONT).pack(side="left")

        self.refresh_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(bar, text="强制刷新数据", variable=self.refresh_var).pack(side="left", padx=12)

        self.prog = ttk.Progressbar(bar, mode="indeterminate", length=130, maximum=100)
        self.prog.pack(side="left", padx=12)

        self.status = tk.Label(bar, text="● 就绪。点击「开始分析」", bg=BG_MAIN,
                               fg=MUTED, font=FONT)
        self.status.pack(side="left", padx=10)

        # 标签页
        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill="both", expand=True, padx=12, pady=(6, 12))

        self.tabs = {}
        titles = [("overview", "总览"), ("full", "三组完整预测"),
                  ("single", "五组单颗特码"), ("dims", "六维度属性"),
                  ("pools", "号码集合"), ("zodiac", "三生肖")]
        for key, title in titles:
            page = tk.Frame(self.nb, bg=BG_MAIN)
            self.nb.add(page, text=title)
            txt = self._make_card(page)
            self.tabs[key] = txt

        self._fill("overview", [("澳门六合彩开奖记录分析与预测\n", "h"),
                                ("\n点击右上方「▶ 开始分析」启动。\n", None),
                                ("程序自动抓取 macaujc.com 历史开奖, 采用统计模型 + 大模型反推理,\n", None),
                                ("给出三组完整预测、五组特码、六维度属性、号码集合、三生肖等多维度预测。\n\n", None),
                                (DISCLAIMER, "muted")])

    def _make_card(self, parent):
        """带边框的卡片文本区。"""
        wrap = tk.Frame(parent, bg=BORDER)
        wrap.pack(fill="both", expand=True, padx=2, pady=2)
        txt = scrolledtext.ScrolledText(wrap, wrap="word", font=FONT, spacing1=2, spacing2=2,
                                        bg=BG_CARD, fg=TEXT, relief="flat", bd=0,
                                        padx=16, pady=12, highlightthickness=0,
                                        insertbackground=PRIMARY, selectbackground=PRIMARY_D)
        txt.pack(fill="both", expand=True, padx=1, pady=1)
        # 文本标签样式
        txt.tag_config("h", font=FONT_H, foreground=PRIMARY, spacing3=4)
        txt.tag_config("star", foreground=ACCENT, font=("Microsoft YaHei UI", 10, "bold"))
        txt.tag_config("muted", foreground=MUTED)
        txt.tag_config("ok", foreground=OK)
        txt.tag_config("warn", foreground=WARN)
        txt.tag_config("mono", font=FONT_MONO)
        txt.tag_config("key", foreground=PRIMARY_L)
        txt.tag_config("chip", font=FONT_MONO_B, background=PRIMARY, foreground="#1a1a22")
        txt.tag_config("chipm", font=FONT_MONO_B, background=BG_CARD2, foreground=PRIMARY_L)
        # 滚动条配色
        try:
            txt.vbar.config(bg=BG_CARD2, troughcolor=BG_CARD,
                            activebackground=PRIMARY_D, highlightthickness=0, bd=0)
        except Exception:
            pass
        txt.configure(state="disabled")
        return txt

    def _add_hover(self, btn, normal, hover):
        def enter(e):
            btn.config(bg=hover)
        def leave(e):
            btn.config(bg=normal)
        btn.bind("<Enter>", enter)
        btn.bind("<Leave>", leave)

    # ---------------- 运行 ----------------
    def start(self):
        if self.worker and self.worker.is_alive():
            return
        self.run_btn.config(state="disabled")
        self.prog.start(12)
        self._set_status("运行中...", PRIMARY)
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

    def _set_status(self, text, color=MUTED):
        self.status.config(text="● " + text, fg=color)

    def _poll(self):
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                if kind == "msg":
                    self._set_status(payload, PRIMARY)
                elif kind == "error":
                    self.prog.stop()
                    self.run_btn.config(state="normal")
                    self._set_status("错误", ACCENT)
                    messagebox.showerror("错误", payload)
                elif kind == "done":
                    self.results = payload
                    self._render_all()
                    self.prog.stop()
                    self.run_btn.config(state="normal")
                    self._set_status("完成", OK)
                    self.nb.select(self.tabs["overview"])
        except queue.Empty:
            pass
        self.root.after(200, self._poll)

    # ---------------- 渲染 ----------------
    def _fill(self, key, segments):
        t = self.tabs[key]
        t.configure(state="normal")
        t.delete("1.0", "end")
        for text, tag in segments:
            if tag:
                t.insert("end", text, tag)
            else:
                t.insert("end", text)
        t.configure(state="disabled")
        t.see("1.0")

    def _lift_tag(self, lift):
        if lift > 1e-9:
            return "ok"
        if lift < -1e-9:
            return "warn"
        return "muted"

    def _render_all(self):
        r = self.results
        recs = r["records"]
        report = r["report"]
        last = recs[-1]

        # ---- 总览 ----
        seg = [("数据概览\n", "h")]
        seg.append((f"期数 ", None)); seg.append((f"{len(recs)}", "key"))
        seg.append((f"     时间范围 ", None))
        seg.append((f"{recs[0].open_time} ~ {last.open_time}\n", "key"))
        seg.append(("最近一期: 期号 ", None))
        seg.append((f"{last.expect}", "key"))
        seg.append(("  平码 ", None))
        seg.append((f" {_fmt_nums(last.regular)} ", "chipm"))
        seg.append(("  特码 ", None))
        seg.append((f" {last.special:02d} ", "chip"))
        seg.append(("\n\n", None))
        seg.append(("反推理候选数学模型\n", "h"))
        seg.append(("(拟合度 0–1, 越高越可能解释该序列)\n\n", "muted"))
        for name, score, detail in report["models"]:
            seg.append((f"  {name}  ", None))
            seg.append((f"{score:.3f}", "key"))
            seg.append((f"   {detail}\n", "muted"))
        llm = r["llm"]
        seg.append(("\n大模型反推理\n", "h"))
        if llm:
            seg.append(("  模型清单: ", None))
            seg.append((f"{', '.join(llm.inferred_models) or '(未解析)'}\n", "key"))
            seg.append((f"  推理: {llm.reasoning[:240]}\n", "muted"))
        else:
            seg.append(("  未启用(编辑 config.ini 填入 api_key)\n", "muted"))
        seg.append(("\n" + DISCLAIMER, "muted"))
        self._fill("overview", seg)

        # ---- 三组完整预测 ----
        seg = [("三组完整预测  (6平码 + 1特码)\n", "h"),
               ("按回测综合命中率排序, ★ 为推荐组\n\n", "muted")]
        for i, g in enumerate(r["groups"]):
            star = " ★ 推荐" if i == 0 else ""
            seg.append((f"{g.name}组{star}:  平码 ", "star" if i == 0 else None))
            seg.append((f" {_fmt_nums(g.regular)} ", "chipm" if i else "chipm"))
            seg.append(("  特码 ", "star" if i == 0 else None))
            seg.append((f" {g.special:02d} ", "chip"))
            seg.append(("\n", None))
            seg.append((f"  策略: {g.strategy}\n", "muted"))
            seg.append((f"  回测: 特码命中率 ", None))
            seg.append((f"{g.backtest_special_hit:.1%}", "key"))
            seg.append((f"   平均平码命中 ", None))
            seg.append((f"{g.backtest_regular_hits:.2f}/6\n\n", "key"))
        self._fill("full", seg)

        # ---- 五组单颗特码 ----
        seg = [("五组单颗特码预测\n", "h"),
               ("随机基线 = 1/49 ≈ ", "muted"), ("2.0%", "key"),
               ("    按回测命中率排序, ★ 为推荐组\n\n", "muted")]
        for i, g in enumerate(r["sp"]):
            star = " ★ 推荐" if i == 0 else ""
            seg.append((f"特码{g.name}组{star}:  特码 ", "star" if i == 0 else None))
            seg.append((f" {g.special:02d} ", "chip"))
            seg.append(("\n", None))
            seg.append((f"  策略: {g.strategy}\n", "muted"))
            seg.append((f"  回测: 命中率 ", None))
            seg.append((f"{g.backtest_hit:.1%}", "key"))
            seg.append((f"   lift ", None))
            seg.append((f"{g.lift:+.1%}", self._lift_tag(g.lift)))
            seg.append((f"   标准误 {g.std_error:.1%}   稳定性 {g.stability:.2f}\n\n", "muted"))
        self._fill("single", seg)

        # ---- 六维度属性 ----
        seg = [("六维度特码属性预测\n", "h"),
               ("频率 + 马尔可夫 + 趋势 融合    lift>0 优于随机基线\n\n", "muted"),
               (f"  {'维度':<6}{'预测值':<8}{'准确率':>8}{'随机基线':>10}{'lift':>9}{'标准误':>8}{'稳定性':>8}\n", "mono")]
        for d in r["dims"]:
            seg.append((f"  {d.name:<6}{d.value:<8}{d.accuracy:>8.1%}{d.baseline:>10.1%}"
                        f"{d.lift:>+9.1%}{d.std_error:>8.1%}{d.stability:>8.2f}\n", "mono"))
        self._fill("dims", seg)

        # ---- 号码集合 ----
        seg = [("特码号码集合预测  (每集合 8~10 颗)\n", "h"),
               ("命中 = 真实特码落在集合内    按回测命中率排序, ★ 为推荐组\n\n", "muted")]
        for i, p in enumerate(r["pools"]):
            star = " ★ 推荐" if i == 0 else ""
            seg.append((f"集合{p.name}组{star} ({len(p.numbers)}颗):  ", "star" if i == 0 else None))
            seg.append((f" {_fmt_nums(p.numbers)} ", "chipm"))
            seg.append(("\n", None))
            seg.append((f"  策略: {p.strategy}\n", "muted"))
            seg.append((f"  回测: 命中率 ", None))
            seg.append((f"{p.hit_rate:.1%}", "key"))
            seg.append((f"   基线 {p.baseline:.1%}   lift ", None))
            seg.append((f"{p.lift:+.1%}", self._lift_tag(p.lift)))
            seg.append((f"   稳定性 {p.stability:.2f}\n\n", "muted"))
        wide = r["wide"]
        seg.append(("20 颗特码大集合  (单组 · 多信号融合)\n", "h"))
        seg.append((f"  ★ 大集合 ({len(wide.numbers)}颗):  ", "star"))
        seg.append((f" {_fmt_nums(wide.numbers)} ", "chip"))
        seg.append(("\n", None))
        seg.append((f"  策略: {wide.strategy}\n", "muted"))
        seg.append((f"  回测: 命中率 ", None))
        seg.append((f"{wide.hit_rate:.1%}", "key"))
        seg.append((f"   基线 {wide.baseline:.1%}   lift ", None))
        seg.append((f"{wide.lift:+.1%}", self._lift_tag(wide.lift)))
        seg.append((f"   稳定性 {wide.stability:.2f}\n", None))
        self._fill("pools", seg)

        # ---- 三生肖 ----
        zp = r["zp"]
        seg = [("特码三生肖预测  (单组)\n", "h"),
               ("命中 = 真实特码生肖落在 3 个内    随机基线 = 3/12 = ", "muted"),
               ("25%\n\n", "key")]
        seg.append(("  ★ 三生肖:  ", "star"))
        for i, z in enumerate(zp.zodiacs):
            seg.append((f" {z} ", "chip"))
            if i < len(zp.zodiacs) - 1:
                seg.append(("  ", None))
        seg.append(("\n", None))
        seg.append((f"  策略: {zp.strategy}\n", "muted"))
        seg.append((f"  回测: 命中率 ", None))
        seg.append((f"{zp.hit_rate:.1%}", "key"))
        seg.append((f"   基线 {zp.baseline:.1%}   lift ", None))
        seg.append((f"{zp.lift:+.1%}", self._lift_tag(zp.lift)))
        seg.append((f"   标准误 {zp.std_error:.1%}   稳定性 {zp.stability:.2f}\n\n", None))
        seg.append((DISCLAIMER, "muted"))
        self._fill("zodiac", seg)


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
