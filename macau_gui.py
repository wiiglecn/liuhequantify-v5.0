#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""澳门六合彩分析与预测 - 图形界面版(tkinter)
把所有预测信息排版布局在标签页界面上。"""
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

FONT = ("Microsoft YaHei UI", 10)
FONT_H = ("Microsoft YaHei UI", 12, "bold")
FONT_MONO = ("Consolas", 10)

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
        root.geometry("980x680")
        self._build_ui()
        self._poll()

    # ---------------- UI 构建 ----------------
    def _build_ui(self):
        bar = ttk.Frame(self.root)
        bar.pack(fill="x", padx=8, pady=6)

        self.run_btn = ttk.Button(bar, text="开始分析", command=self.start)
        self.run_btn.pack(side="left")

        ttk.Label(bar, text="  回测期数:").pack(side="left")
        self.bt_var = tk.IntVar(value=30)
        ttk.Spinbox(bar, from_=5, to=100, textvariable=self.bt_var,
                    width=5).pack(side="left", padx=4)

        self.refresh_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(bar, text="强制刷新数据", variable=self.refresh_var).pack(side="left", padx=8)

        self.prog = ttk.Progressbar(bar, mode="indeterminate", length=120)
        self.prog.pack(side="left", padx=8)

        self.status = ttk.Label(bar, text="就绪。点击「开始分析」", font=FONT)
        self.status.pack(side="left", padx=10)

        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill="both", expand=True, padx=8, pady=4)

        self.tabs = {}
        titles = [("overview", "总览"), ("full", "三组完整预测"),
                  ("single", "五组单颗特码"), ("dims", "六维度属性"),
                  ("pools", "号码集合"), ("zodiac", "三生肖")]
        for key, title in titles:
            txt = scrolledtext.ScrolledText(self.nb, wrap="word", font=FONT, spacing1=2)
            txt.tag_config("h", font=FONT_H, foreground="#1a4f8b")
            txt.tag_config("star", foreground="#c0392b", font=("Microsoft YaHei UI", 10, "bold"))
            txt.tag_config("muted", foreground="#777")
            txt.tag_config("ok", foreground="#1e8449")
            txt.tag_config("mono", font=FONT_MONO)
            txt.configure(state="disabled")
            self.nb.add(txt, text=title)
            self.tabs[key] = txt

        self._fill("overview", [("澳门六合彩开奖记录分析与预测软件\n", "h"),
                                ("\n点击上方「开始分析」按钮开始。\n", None),
                                ("\n说明: 程序会自动抓取 macaujc.com 历史开奖记录,\n", None),
                                ("采用统计模型 + 大模型反推理, 给出多维度预测。\n\n", None),
                                (DISCLAIMER, "muted")])

    # ---------------- 运行 ----------------
    def start(self):
        if self.worker and self.worker.is_alive():
            return
        self.run_btn.config(state="disabled")
        self.prog.start(12)
        self.status.config(text="分析中...")
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

    def _poll(self):
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                if kind == "msg":
                    self.status.config(text=payload)
                elif kind == "error":
                    self.prog.stop()
                    self.run_btn.config(state="normal")
                    self.status.config(text="错误: " + payload)
                    messagebox.showerror("错误", payload)
                elif kind == "done":
                    self.results = payload
                    self._render_all()
                    self.prog.stop()
                    self.run_btn.config(state="normal")
                    self.status.config(text="完成。")
                    self.nb.select(self.tabs["overview"])
        except queue.Empty:
            pass
        self.root.after(200, self._poll)

    # ---------------- 渲染 ----------------
    def _fill(self, key, segments):
        """segments: list of (text, tag|None)"""
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

    def _render_all(self):
        r = self.results
        recs = r["records"]
        report = r["report"]
        last = recs[-1]

        # ---- 总览 ----
        seg = [("数据概览\n", "h"),
               (f"期数: {len(recs)}    时间范围: {recs[0].open_time} ~ {last.open_time}\n", None),
               (f"最近一期: 期号 {last.expect}  平码 {_fmt_nums(last.regular)}  特码 {last.special:02d}\n\n", None),
               ("反推理候选数学模型(拟合度 0-1, 越高越可能解释该序列)\n", "h")]
        for name, score, detail in report["models"]:
            seg.append((f"  {name}: {score:.3f}  ({detail})\n", None))
        llm = r["llm"]
        seg.append(("\n大模型反推理\n", "h"))
        if llm:
            seg.append((f"  模型清单: {', '.join(llm.inferred_models) or '(未解析)'}\n", None))
            seg.append((f"  推理: {llm.reasoning[:200]}\n", "muted"))
        else:
            seg.append(("  未启用(编辑 config.ini 填入 api_key)\n", "muted"))
        seg.append(("\n" + DISCLAIMER + "\n", "muted"))
        self._fill("overview", seg)

        # ---- 三组完整预测 ----
        seg = [("三组完整预测(6平码 + 1特码)\n", "h"),
               ("按回测综合命中率排序, ★ 为推荐组\n\n", "muted")]
        for i, g in enumerate(r["groups"]):
            star = " ★推荐" if i == 0 else ""
            seg.append((f"{g.name}组{star}: 平码 {_fmt_nums(g.regular)}  特码 {g.special:02d}\n",
                        "star" if i == 0 else None))
            seg.append((f"  策略: {g.strategy}\n", "muted"))
            seg.append((f"  回测: 特码命中率={g.backtest_special_hit:.1%}"
                        f"  平均平码命中={g.backtest_regular_hits:.2f}/6\n\n", None))
        self._fill("full", seg)

        # ---- 五组单颗特码 ----
        seg = [("五组单颗特码预测(随机基线 = 1/49 ≈ 2.0%)\n", "h"),
               ("按回测命中率排序, ★ 为推荐组\n\n", "muted")]
        for i, g in enumerate(r["sp"]):
            star = " ★推荐" if i == 0 else ""
            seg.append((f"特码{g.name}组{star}: 特码 {g.special:02d}\n",
                        "star" if i == 0 else None))
            seg.append((f"  策略: {g.strategy}\n", "muted"))
            seg.append((f"  回测: 命中率={g.backtest_hit:.1%}  lift={g.lift:+.1%}"
                        f"  标准误={g.std_error:.1%}  稳定性={g.stability:.2f}\n\n", None))
        self._fill("single", seg)

        # ---- 六维度属性 ----
        seg = [("六维度特码属性预测(频率+马尔可夫+趋势 融合)\n", "h"),
               ("lift>0 表示优于随机基线; 稳定性越小越稳\n\n", "muted"),
               (f"  {'维度':<6}{'预测值':<8}{'准确率':>8}{'随机基线':>10}{'lift':>8}{'标准误':>8}{'稳定性':>8}\n", "mono")]
        for d in r["dims"]:
            seg.append((f"  {d.name:<6}{d.value:<8}{d.accuracy:>8.1%}{d.baseline:>10.1%}"
                        f"{d.lift:>+8.1%}{d.std_error:>8.1%}{d.stability:>8.2f}\n", "mono"))
        self._fill("dims", seg)

        # ---- 号码集合 ----
        seg = [("特码号码集合预测(每集合 8~10 颗, 命中=真实特码落在集合内)\n", "h"),
               ("按回测命中率排序, ★ 为推荐组\n\n", "muted")]
        for i, p in enumerate(r["pools"]):
            star = " ★推荐" if i == 0 else ""
            seg.append((f"集合{p.name}组{star} ({len(p.numbers)}颗): {_fmt_nums(p.numbers)}\n",
                        "star" if i == 0 else None))
            seg.append((f"  策略: {p.strategy}\n", "muted"))
            seg.append((f"  回测: 命中率={p.hit_rate:.1%}  基线={p.baseline:.1%}"
                        f"  lift={p.lift:+.1%}  稳定性={p.stability:.2f}\n\n", None))
        wide = r["wide"]
        seg.append(("20 颗特码大集合(单组, 多信号融合)\n", "h"))
        seg.append((f"  ★大集合 ({len(wide.numbers)}颗): {_fmt_nums(wide.numbers)}\n", "star"))
        seg.append((f"  策略: {wide.strategy}\n", "muted"))
        seg.append((f"  回测: 命中率={wide.hit_rate:.1%}  基线={wide.baseline:.1%}"
                    f"  lift={wide.lift:+.1%}  稳定性={wide.stability:.2f}\n", None))
        self._fill("pools", seg)

        # ---- 三生肖 ----
        zp = r["zp"]
        seg = [("特码三生肖预测(单组, 命中=真实特码生肖落在 3 个内)\n", "h"),
               ("随机基线 = 3/12 = 25%\n\n", "muted"),
               (f"  ★三生肖: {'、'.join(zp.zodiacs)}\n", "star"),
               (f"  策略: {zp.strategy}\n", "muted"),
               (f"  回测: 命中率={zp.hit_rate:.1%}  基线={zp.baseline:.1%}"
                 f"  lift={zp.lift:+.1%}  标准误={zp.std_error:.1%}  稳定性={zp.stability:.2f}\n\n", None),
               (DISCLAIMER + "\n", "muted")]
        self._fill("zodiac", seg)


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
