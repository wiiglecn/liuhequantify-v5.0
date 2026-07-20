# macau_gui.py 界面重设计实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 macau_gui.py 改造为「浅色现代分析台 × 左侧导航」风格(设计文档: `docs/superpowers/specs/2026-07-20-gui-redesign-design.md`),预测/数据/存档逻辑零改动。

**Architecture:** 单文件改造。新增 3 个纯绘制助手(PIL 渐变/柔影、矢量圆角矩形)+ 3 个控件类(RoundedCard/RoundedButton/_NavItem)+ 2 个数据模型(NAV_GROUPS/PAGE_TITLES);`_build_ui` 重写为 侧导航+顶栏+渐变横幅+页栈,替换 ttk.Notebook;10 个页面渲染函数复用,仅换容器与令牌。无 PIL 时全面降级(矢量圆角/纯色)。

**Tech Stack:** Python 3.13 / tkinter(ttk clam)/ Pillow+numpy(可选,沿用现有 `_PIL_OK` 降级模式)/ pytest。

**执行环境注意(务必遵守):**
- 本机 Bash 跑 python/pytest 可能被 auto 权限模式拦截 → 若被拦,请用户在终端用 `! python -m pytest ...` 运行并贴回输出,不要反复重试 Bash。
- `macau_gui.py` 字符串占位用 em dash(U+2014 `—`),缩进为空格;Edit 的 old_string 必须按 Read 原文精确匹配(含 `—`)。
- 现有 pytest 套件(tests/test_dimensions.py 等)不导入 GUI,必须保持全绿。

**基线提交:** `77a773a`(设计文档提交)。Codex 审查 diff 范围以此为准。

---

## 文件结构

| 文件 | 动作 | 职责 |
|---|---|---|
| `macau_gui.py` | 修改 | 全部:令牌、助手、控件类、布局、页面渲染换肤 |
| `tests/test_gui_theme.py` | 新建 | 令牌/助手/导航模型单测 + App 构建冒烟测试(Tk 不可用自动 skip) |

---

### Task 1: 主题令牌 + 纯绘制助手(渐变/柔影/圆角)

**Files:**
- Modify: `macau_gui.py`(docstring 第 3-4 行;令牌块第 44-67 行;`_render_ball_pil` 与 `draw_sphere` 的 `SHADOW` 引用)
- Test: `tests/test_gui_theme.py`

- [ ] **Step 1: 写失败测试(令牌 + 渐变/柔影助手部分)**

新建 `tests/test_gui_theme.py`:

```python
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_gui_theme.py -v`
Expected: FAIL — `AttributeError: module 'macau_gui' has no attribute '_gradient_pil'`(及 BG_SIDE/BALL_SHADOW 等缺失)

- [ ] **Step 3: 替换令牌块 + 新增助手**

(a) docstring 第 3-4 行改为:

```python
"""澳门六合彩分析与预测 - 图形界面版(tkinter)
浅色现代分析台主题 + 左侧导航控制台 + 3D 立体号码球 + 圆角卡片流。"""
```

(b) 令牌块(现第 44-67 行,`# ======================== 主题配色...` 至 `FONT_ZOD = ...`)整体替换为:

```python
# ======================== 主题配色 (浅色现代分析台) ========================
BG_MAIN = "#eef1f6"        # 页面背景(浅灰蓝)
BG_CARD = "#ffffff"        # 卡片背景(白)
BG_CARD2 = "#f8fafc"       # 嵌套层/磁贴底
BG_SIDE = "#ffffff"        # 侧边导航底
BORDER = "#edf0f5"         # 卡片边框
BORDER_G = "#e3e7ef"       # 通用边框
SHADOW = "#1a202c"         # 投影基色(PIL 低透明度柔影用)
BALL_SHADOW = "#c9d1de"    # 号码球投影(浅底适配)
PRIMARY = "#e11d48"        # 主色(红): 焦点/推荐/主按钮
PRIMARY_D = "#be123c"      # 深红(悬停)
PRIMARY_L = "#fb7185"      # 浅红(渐变端点)
PRIMARY_BG = "#fff1f2"     # 主色浅底(导航选中/胶囊)
ACCENT = "#e11d48"         # 高亮(同主色)
INDIGO = "#6366f1"         # 辅助靛蓝(数据第二系)
AMBER = "#f59e0b"          # 数据第三系
TEXT = "#1a202c"           # 主文字
TEXT2 = "#4b5261"          # 次文字
MUTED = "#9aa1ad"          # 弱化文字(柔和灰)
OK = "#059669"             # 正向/命中/就绪
WARN = "#d97706"           # 警示/负向
OK_BG = "#f0fdf4"          # 状态卡浅绿底
OK_BORDER = "#d1fae5"      # 状态卡边框
TRACK = "#eef1f6"          # 进度条/数据条轨道

FONT = ("Microsoft YaHei UI", 10)
FONT_SM = ("Microsoft YaHei UI", 9)
FONT_H = ("Microsoft YaHei UI", 12, "bold")
FONT_TITLE = ("Microsoft YaHei UI", 18, "bold")
FONT_NUM = ("Consolas", 12, "bold")
FONT_NUM_S = ("Consolas", 11, "bold")
FONT_ZOD = ("Microsoft YaHei UI", 12, "bold")
FONT_NAV = ("Microsoft YaHei UI", 10)
FONT_NAV_B = ("Microsoft YaHei UI", 10, "bold")
```

(c) 球体投影适配浅底 — 两处 `SHADOW` 引用改 `BALL_SHADOW`:
- `_render_ball_pil` 中 `fill=(*_hex_rgb(SHADOW), 90))` → `fill=(*_hex_rgb(BALL_SHADOW), 90))`
- `draw_sphere` 中 `fill=SHADOW, outline=""` → `fill=BALL_SHADOW, outline=""`;`outline=_blend(base, SHADOW, 0.45)` → `outline=_blend(base, BALL_SHADOW, 0.45)`

(d) 在 `_hex_rgb` 函数之后追加三个纯绘制助手:

```python
def _gradient_pil(w, h, c1, c2, radius=0):
    """PIL 渲染水平双色渐变圆角图(RGBA); 无 PIL 返回 None。

    横幅/Logo/数据条用; 水平渐变横向拉伸不失真, 可缓存复用。
    """
    if not _PIL_OK:
        return None
    w, h = max(2, int(w)), max(2, int(h))
    r1, g1, b1 = _hex_rgb(c1)
    r2, g2, b2 = _hex_rgb(c2)
    t = _np.linspace(0.0, 1.0, w, dtype=_np.float32)
    row = _np.stack([r1 + (r2 - r1) * t,
                     g1 + (g2 - g1) * t,
                     b1 + (b2 - b1) * t], axis=-1)
    arr = _np.repeat(row[_np.newaxis, :, :], h, axis=0).clip(0, 255).astype(_np.uint8)
    alpha = _np.full((h, w), 255, dtype=_np.uint8)
    if radius > 0:
        mask = Image.new("L", (w, h), 0)
        ImageDraw.Draw(mask).rounded_rectangle([0, 0, w - 1, h - 1],
                                               radius=radius, fill=255)
        alpha = _np.array(mask)
    return Image.fromarray(_np.dstack([arr, alpha]), "RGBA")


def _shadow_pil(w, h, radius, color=SHADOW, alpha=20, blur=8, dy=3):
    """圆角卡片柔影(RGBA 透明底): 同形深色斑 + 高斯柔化 + 向下偏移 dy。无 PIL 返回 None。"""
    if not _PIL_OK:
        return None
    w, h = max(2, int(w)), max(2, int(h))
    m = blur + 4
    sh = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(sh).rounded_rectangle(
        [m, m + dy, w - m, h - m + dy], radius=radius,
        fill=(*_hex_rgb(color), alpha))
    return sh.filter(ImageFilter.GaussianBlur(blur))


def draw_round_rect(cv, x1, y1, x2, y2, r, fill=None, outline=None, width=1,
                    dash=None, tags=None):
    """在 Canvas 上画圆角矩形(4 弧 + 2 矩形 / 4 弧 + 4 线, 纯矢量, 任意缩放不失真)。

    dash 非空时边框为虚线(科学提示卡); tags 传给各图元便于批量删除/升降层。
    """
    r = max(0.0, min(float(r), (x2 - x1) / 2.0, (y2 - y1) / 2.0))
    if r <= 0:
        if fill:
            cv.create_rectangle(x1, y1, x2, y2, fill=fill, outline="", tags=tags)
        if outline:
            cv.create_rectangle(x1, y1, x2, y2, outline=outline, width=width,
                                dash=dash, tags=tags)
        return
    corners = (((x1, y1, x1 + 2 * r, y1 + 2 * r), 90),
               ((x2 - 2 * r, y1, x2, y1 + 2 * r), 0),
               ((x1, y2 - 2 * r, x1 + 2 * r, y2), 180),
               ((x2 - 2 * r, y2 - 2 * r, x2, y2), 270))
    if fill:
        for box, start in corners:
            cv.create_arc(*box, start=start, extent=90, style="pieslice",
                          fill=fill, outline="", tags=tags)
        cv.create_rectangle(x1 + r, y1, x2 - r, y2, fill=fill, outline="", tags=tags)
        cv.create_rectangle(x1, y1 + r, x2, y2 - r, fill=fill, outline="", tags=tags)
    if outline:
        for box, start in corners:
            cv.create_arc(*box, start=start, extent=90, style="arc",
                          outline=outline, width=width, dash=dash, tags=tags)
        cv.create_line(x1 + r, y1, x2 - r, y1, fill=outline, width=width, dash=dash, tags=tags)
        cv.create_line(x1 + r, y2, x2 - r, y2, fill=outline, width=width, dash=dash, tags=tags)
        cv.create_line(x1, y1 + r, x1, y2 - r, fill=outline, width=width, dash=dash, tags=tags)
        cv.create_line(x2, y1 + r, x2, y2 - r, fill=outline, width=width, dash=dash, tags=tags)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_gui_theme.py -v`
Expected: 4 passed(skipif 视 PIL 而定,本机有 PIL 应全过)

- [ ] **Step 5: 运行既有套件确认未破坏**

Run: `python -m pytest -q`
Expected: 全绿(其余测试不导入 macau_gui,仅确认 import 无副作用)

- [ ] **Step 6: Commit**

```bash
git add macau_gui.py tests/test_gui_theme.py
git commit -m "feat(gui): 浅色现代主题令牌 + 渐变/柔影/圆角绘制助手"
```

### Task 2: 圆角控件类(RoundedCard / RoundedButton / 胶囊)

**Files:**
- Modify: `macau_gui.py`(`draw_round_rect` 之后插入控件类;`_significance_chip` 改胶囊)
- Test: `tests/test_gui_theme.py`(追加)

- [ ] **Step 1: 追加失败测试(控件冒烟 + 胶囊/卡片)**

在 `tests/test_gui_theme.py` 末尾追加:

```python
def _tk_or_skip():
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("无显示环境")
    root.withdraw()
    return root


def test_draw_round_rect_smoke():
    root = _tk_or_skip()
    tk = pytest.importorskip("tkinter")
    cv = tk.Canvas(root, width=60, height=40)
    mg.draw_round_rect(cv, 2, 2, 58, 38, 10, fill="#ffffff", outline="#edf0f5")
    assert len(cv.find_all()) >= 6          # 4 弧 + 2 矩形(+ 边框图元)
    mg.draw_round_rect(cv, 2, 2, 58, 38, 0, fill="#ffffff")   # r=0 退化矩形
    root.destroy()


def test_rounded_card_hosts_body():
    root = _tk_or_skip()
    card = mg.RoundedCard(root, radius=12)
    card.pack()
    tk = pytest.importorskip("tkinter")
    tk.Label(card.body, text="hi", bg=mg.BG_CARD).pack()
    root.update_idletasks()
    assert card.body.winfo_width() > 0
    root.destroy()


def test_rounded_button_states():
    root = _tk_or_skip()
    clicks = []
    btn = mg.RoundedButton(root, "▶ 开始分析", command=lambda: clicks.append(1))
    btn.pack()
    btn._on_click(None)
    assert clicks == [1]
    btn.set_enabled(False)
    btn._on_click(None)
    assert clicks == [1]                    # 禁用后不再触发
    root.destroy()


def test_make_capsule_size():
    root = _tk_or_skip()
    cap = mg.make_capsule(root, "★ 推荐", mg.PRIMARY_BG, fg=mg.PRIMARY)
    assert int(cap.cget("width")) > 10 and int(cap.cget("height")) > 10
    root.destroy()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_gui_theme.py -v`
Expected: FAIL — `AttributeError: module 'macau_gui' has no attribute 'RoundedCard'`

- [ ] **Step 3: 实现控件类**

在 `draw_round_rect` 之后插入:

```python
class RoundedCard(tk.Canvas):
    """圆角卡片容器: 矢量圆角矩形 + 可选 PIL 柔影; 内容放进 .body Frame。

    - 圆角纯矢量(无 PIL 也能工作); 柔影为 PIL 高斯图(无 PIL 自动跳过)
    - body 内缩 pad ≥ 半径, 不遮圆角; 宽度随容器拉伸, 高度自适应内容
    - dash 非空时边框为虚线(科学提示卡用)
    """

    def __init__(self, parent, radius=12, fill=BG_CARD, border=BORDER,
                 shadow=True, pad=None, dash=None):
        super().__init__(parent, bg=parent["bg"], highlightthickness=0, bd=0)
        self._r, self._fill, self._border = radius, fill, border
        self._shadow_on = shadow and _PIL_OK
        self._dash = dash
        self._pad = pad if pad is not None else radius + 4
        self._shadow_img = None
        self.body = tk.Frame(self, bg=fill)
        self._win = None
        self.body.bind("<Configure>", lambda _e: self._fit())
        self.bind("<Configure>", lambda _e: self._redraw())
        self._fit()

    def _fit(self):
        w = self.body.winfo_reqwidth() + self._pad * 2
        h = self.body.winfo_reqheight() + self._pad * 2
        if (w, h) != (self.winfo_width(), self.winfo_height()):
            self.configure(width=w, height=h)

    def _redraw(self):
        w = max(2, self.winfo_width())
        h = max(2, self.winfo_height())
        self.delete("all")
        off = 0
        if self._shadow_on:
            self._shadow_img = ImageTk.PhotoImage(_shadow_pil(w, h, self._r))
            self.create_image(0, 0, image=self._shadow_img, anchor="nw")
            off = 3
        draw_round_rect(self, 0.5, 0.5, w - 1.5 - off, h - 1.5 - off, self._r,
                        fill=self._fill, outline=self._border, dash=self._dash)
        self._win = self.create_window(self._pad, self._pad,
                                       window=self.body, anchor="nw",
                                       width=w - self._pad * 2)
        self.after_idle(self._fit)


class RoundedButton(tk.Canvas):
    """圆角按钮(纯矢量, 无图片依赖)。

    kind: primary(红底白字) / outline(白底红边红字) / ghost(浅灰底) / danger(警示文字)。
    用 set_enabled(bool) 替代 tk.Button 的 state 配置。
    """

    _KINDS = {
        "primary": (PRIMARY, PRIMARY_D, "#ffffff", None),
        "outline": ("#ffffff", PRIMARY_BG, PRIMARY, PRIMARY_L),
        "ghost":   (BG_CARD2, BORDER_G, TEXT2, None),
        "danger":  (BG_CARD2, "#fde8e8", WARN, None),
    }

    def __init__(self, parent, text, command=None, kind="primary",
                 font=("Microsoft YaHei UI", 10, "bold"), padx=18, pady=8):
        super().__init__(parent, bg=parent["bg"], highlightthickness=0, bd=0,
                         cursor="hand2")
        self._cmd = command
        self._enabled = True
        self._hover = False
        self._bg, self._bgh, self._fg, self._border = self._KINDS[kind]
        self._padx, self._pady = padx, pady
        self._tid = self.create_text(padx, pady, text=text, fill=self._fg,
                                     font=font, anchor="nw")
        bx1, by1, bx2, by2 = self.bbox(self._tid)
        self._w = (bx2 - bx1) + padx * 2
        self._h = (by2 - by1) + pady * 2
        self.configure(width=self._w, height=self._h)
        self._paint()
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_click)

    def _paint(self):
        self.delete("shape")
        if not self._enabled:
            fill, fg = "#e5e9f0", "#9aa1ad"
        else:
            fill = self._bgh if self._hover else self._bg
            fg = self._fg
        draw_round_rect(self, 0.5, 0.5, self._w - 1, self._h - 1, 8,
                        fill=fill, outline=self._border, tags="shape")
        self.itemconfigure(self._tid, fill=fg)
        self.tag_raise(self._tid)

    def _on_enter(self, _e):
        self._hover = True
        self._paint()

    def _on_leave(self, _e):
        self._hover = False
        self._paint()

    def _on_click(self, _e):
        if self._enabled and self._cmd:
            self._cmd()

    def set_enabled(self, on):
        self._enabled = bool(on)
        self.configure(cursor="hand2" if on else "arrow")
        self._paint()


def make_capsule(parent, text, bg, fg="#ffffff",
                 font=("Microsoft YaHei UI", 9, "bold"), padx=12, pady=4):
    """圆角胶囊标签(显著性色块/★推荐), 返回未布局的 Canvas。"""
    cv = tk.Canvas(parent, bg=parent["bg"], highlightthickness=0, bd=0)
    tid = cv.create_text(padx, pady, text=text, fill=fg, font=font, anchor="nw")
    bx1, by1, bx2, by2 = cv.bbox(tid)
    w, h = bx2 + padx, by2 + pady
    cv.configure(width=w, height=h)
    draw_round_rect(cv, 0.5, 0.5, w - 1, h - 1, h / 2, fill=bg)
    cv.tag_raise(tid)
    return cv
```

随后把 `_significance_chip` 整体替换为胶囊实现(签名与返回值不变):

```python
    def _significance_chip(self, parent, lift, std_error):
        """提升显著性胶囊: 绿=lift>标准误(有效), 灰=噪声, 橙=显著低于基线。"""
        _short, full, color = self._sig_verdict(lift, std_error)
        try:
            se = float(std_error or 0.0)
        except (TypeError, ValueError):
            se = 0.0
        cap = make_capsule(
            parent,
            f" 提升显著性  {full}   提升 {lift:+.1%}   标准误 {se:.1%} ",
            color)
        cap.pack(anchor="w", pady=(4, 2))
        return cap
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_gui_theme.py -v`
Expected: 全部通过(Tk 冒烟测试本机可跑;无显示环境自动 skip)

- [ ] **Step 5: Commit**

```bash
git add macau_gui.py tests/test_gui_theme.py
git commit -m "feat(gui): 圆角卡片/圆角按钮/胶囊控件 + 显著性色块改胶囊"
```

### Task 3: 侧边导航数据模型(NAV_GROUPS / PAGE_TITLES / _NavItem)

**Files:**
- Modify: `macau_gui.py`(模块级常量区;`_NavItem` 类)
- Test: `tests/test_gui_theme.py`(追加)

- [ ] **Step 1: 追加失败测试**

```python
def test_nav_structure():
    keys = [k for _g, items in mg.NAV_GROUPS for k, _t in items]
    assert len(keys) == len(set(keys)) == 10
    assert set(keys) == {"overview", "full", "single", "dims", "pools",
                         "zodiac", "zodiac_quad", "zodiac_six", "records", "history"}
    assert set(mg.PAGE_TITLES) == set(keys)
    for key, (title, sub) in mg.PAGE_TITLES.items():
        assert title and sub, key


def test_nav_item_selection_paint():
    root = _tk_or_skip()
    picked = []
    it = mg._NavItem(root, "overview", "📊 总览", lambda k: picked.append(k))
    it.pack()
    it.set_selected(True)
    assert it.lbl.cget("fg") == mg.PRIMARY
    it.set_selected(False)
    assert it.lbl.cget("fg") == mg.TEXT2
    it._click(None)
    assert picked == ["overview"]
    root.destroy()
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_gui_theme.py -v -k nav`
Expected: FAIL — `AttributeError: ... 'NAV_GROUPS'`

- [ ] **Step 3: 实现数据模型与导航项类**

在模块级常量区(`DISCLAIMER` 定义之前)插入:

```python
# 侧边导航: (分组名, [(页面key, 导航文案)])
NAV_GROUPS = [
    ("预测分析", [("overview", "📊 总览"),
                  ("full", "🎯 三组完整预测"),
                  ("single", "🔮 五组单颗特码"),
                  ("dims", "📐 六维度属性"),
                  ("pools", "🧮 号码集合")]),
    ("生肖预测", [("zodiac", "🐯 三生肖"),
                  ("zodiac_quad", "🐉 四生肖"),
                  ("zodiac_six", "🐍 六生肖")]),
    ("档案",     [("records", "📜 开奖记录"),
                  ("history", "🗂 历史预测")]),
]

# 顶栏标题: 页面key -> (标题, 副标题)
PAGE_TITLES = {
    "overview":    ("总览", "数据概览与模型状态"),
    "full":        ("三组完整预测", "平码 + 特码 · 三策略并行"),
    "single":      ("五组单颗特码", "单颗特码 · 五策略候选"),
    "dims":        ("六维度属性", "联合一致预测 · 各维度由同一号码导出"),
    "pools":       ("号码集合", "特码集合 · 命中 = 真实特码落在集合内"),
    "zodiac":      ("三生肖", "随机基线 3/12 = 25%"),
    "zodiac_quad": ("四生肖", "随机基线 4/12 ≈ 33.3%"),
    "zodiac_six":  ("六生肖", "随机基线 6/12 = 50%"),
    "records":     ("开奖记录", "历史开奖浏览与单期详情"),
    "history":     ("历史预测", "预测档案与命中核对"),
}
```

在 `make_capsule` 之后插入 `_NavItem`:

```python
class _NavItem(tk.Frame):
    """侧边导航项: 左 3px 指示条 + 文本; 选中浅红底红字加粗, 悬停浅灰。"""

    def __init__(self, parent, key, text, on_select):
        super().__init__(parent, bg=BG_SIDE, cursor="hand2")
        self.key = key
        self._on_select = on_select
        self._selected = False
        self._hover = False
        self.bar = tk.Frame(self, bg=BG_SIDE, width=3)
        self.bar.pack(side="left", fill="y")
        self.lbl = tk.Label(self, text=text, bg=BG_SIDE, fg=TEXT2, font=FONT_NAV,
                            anchor="w", padx=10, pady=7, cursor="hand2")
        self.lbl.pack(side="left", fill="x", expand=True)
        for w in (self, self.lbl):
            w.bind("<Button-1>", self._click)
            w.bind("<Enter>", lambda _e: self._set_hover(True))
            w.bind("<Leave>", lambda _e: self._set_hover(False))

    def _click(self, _e):
        self._on_select(self.key)

    def _set_hover(self, over):
        self._hover = over
        self._paint()

    def set_selected(self, on):
        self._selected = bool(on)
        self._paint()

    def _paint(self):
        if self._selected:
            bg, fg, bar, font = PRIMARY_BG, PRIMARY, PRIMARY, FONT_NAV_B
        elif self._hover:
            bg, fg, bar, font = "#f4f6fa", TEXT, "#f4f6fa", FONT_NAV
        else:
            bg, fg, bar, font = BG_SIDE, TEXT2, BG_SIDE, FONT_NAV
        self.configure(bg=bg)
        self.bar.configure(bg=bar)
        self.lbl.configure(bg=bg, fg=fg, font=font)
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/test_gui_theme.py -v`
Expected: 全部通过

- [ ] **Step 5: Commit**

```bash
git add macau_gui.py tests/test_gui_theme.py
git commit -m "feat(gui): 侧边导航数据模型与导航项控件"
```

### Task 4: 布局骨架重构(侧导航 + 顶栏 + 渐变横幅 + 页栈,替换 Notebook)

**Files:**
- Modify: `macau_gui.py`(`App.__init__`、`_setup_style`、`_build_ui`、`_hover`(删除)、`_set_next_issue`、`_tick_countdown`、`_poll` 中 `self.nb.select(...)`、`_submit_entry` 中 `self.nb.select(self._hist_page)`、`_on_tab_changed`(删除))
- Test: `tests/test_gui_theme.py`(追加 App 冒烟)

- [ ] **Step 1: 追加失败测试(App 构建与页面切换冒烟)**

```python
def test_app_build_and_nav():
    root = _tk_or_skip()
    app = mg.App(root)
    assert len(app._pages) == 10
    assert set(app._pages) == set(mg.PAGE_TITLES)
    for key in app._pages:
        app._show_page(key)
        assert app._current == key
    # 切到历史预测页再切回, 不抛异常即通过(刷新钩子已接线)
    app._show_page("history")
    app._show_page("overview")
    root.destroy()
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_gui_theme.py -v -k app_build`
Expected: FAIL — `AttributeError: 'App' object has no attribute '_pages'`(当前还是 Notebook 结构)

- [ ] **Step 3a: `__init__` 窗口尺寸**

```python
        root.title("澳门六合彩开奖记录分析与预测-猎手2026版")
        root.geometry("1200x800")
        root.minsize(1080, 720)
        root.configure(bg=BG_MAIN)
```

- [ ] **Step 3b: `_setup_style` 整体替换(去掉 Notebook 样式,全面换令牌)**

```python
    def _setup_style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TCheckbutton", background=BG_CARD, foreground=TEXT2, font=FONT_SM)
        style.configure("TSpinbox", fieldbackground="#ffffff", foreground=TEXT,
                        background=BG_CARD, arrowcolor=PRIMARY, bordercolor=BORDER_G,
                        lightcolor=BORDER_G, darkcolor=BORDER_G, insertcolor=PRIMARY)
        style.configure("Horizontal.TProgressbar", background=PRIMARY,
                        troughcolor=TRACK, borderwidth=0,
                        lightcolor=PRIMARY, darkcolor=PRIMARY)
        style.configure("Vertical.TScrollbar", background=BORDER_G,
                        troughcolor=BG_MAIN, borderwidth=0, arrowcolor=TEXT2,
                        gripcount=0)
        style.configure("Treeview", background=BG_CARD, fieldbackground=BG_CARD,
                        foreground=TEXT, borderwidth=0, rowheight=26, font=FONT_SM)
        style.configure("Treeview.Heading", background=BG_CARD2, foreground=TEXT2,
                        font=FONT_SM, borderwidth=0, relief="flat")
        style.map("Treeview", background=[("selected", PRIMARY_BG)],
                  foreground=[("selected", TEXT)])
        style.map("Treeview.Heading", background=[("active", BG_CARD2)])
```

- [ ] **Step 3c: `_build_ui` 整体替换为 侧导航 + 顶栏 + 横幅 + 页栈**

```python
    # ---------------- UI 构建 ----------------
    def _build_ui(self):
        self.bt_var = tk.IntVar(value=60)
        self.refresh_var = tk.BooleanVar(value=False)
        shell = tk.Frame(self.root, bg=BG_MAIN)
        shell.pack(fill="both", expand=True)

        # ---- 侧边导航 ----
        side = tk.Frame(shell, bg=BG_SIDE, width=184,
                        highlightthickness=1, highlightbackground=BORDER)
        side.pack(side="left", fill="y")
        side.pack_propagate(False)
        # 品牌块
        brand = tk.Frame(side, bg=BG_SIDE)
        brand.pack(fill="x", padx=14, pady=(14, 4))
        logo = tk.Canvas(brand, width=32, height=32, bg=BG_SIDE, highlightthickness=0)
        logo.pack(side="left")
        _lg = _gradient_pil(32, 32, PRIMARY, PRIMARY_L, radius=9) if _PIL_OK else None
        if _lg is not None:
            self._logo_img = ImageTk.PhotoImage(_lg)
            logo.create_image(16, 16, image=self._logo_img)
        else:
            draw_round_rect(logo, 1, 1, 31, 31, 9, fill=PRIMARY)
        logo.create_text(16, 16, text="猎", fill="#ffffff",
                         font=("Microsoft YaHei UI", 12, "bold"))
        btxt = tk.Frame(brand, bg=BG_SIDE)
        btxt.pack(side="left", padx=8)
        tk.Label(btxt, text="猎手 2026", bg=BG_SIDE, fg=TEXT,
                 font=("Microsoft YaHei UI", 11, "bold")).pack(anchor="w")
        tk.Label(btxt, text="六合彩分析预测", bg=BG_SIDE, fg=MUTED,
                 font=("Microsoft YaHei UI", 8)).pack(anchor="w")
        # 分组导航
        self._nav_items = {}
        nav = tk.Frame(side, bg=BG_SIDE)
        nav.pack(fill="x", padx=8)
        for group, items in NAV_GROUPS:
            tk.Label(nav, text=group, bg=BG_SIDE, fg="#b6bcc7",
                     font=("Microsoft YaHei UI", 8)).pack(anchor="w", padx=10, pady=(10, 2))
            for key, text in items:
                it = _NavItem(nav, key, text, self._show_page)
                it.pack(fill="x", pady=1)
                self._nav_items[key] = it
        # 底部常驻状态卡
        stat = RoundedCard(side, radius=8, fill=OK_BG, border=OK_BORDER,
                           shadow=False, pad=10)
        stat.pack(side="bottom", fill="x", padx=12, pady=12)
        self.status = tk.Label(stat.body, text="● 就绪。点击「开始分析」",
                               bg=OK_BG, fg=OK, font=FONT_SM,
                               anchor="w", justify="left", wraplength=140)
        self.status.pack(fill="x")

        # ---- 主区 ----
        main = tk.Frame(shell, bg=BG_MAIN)
        main.pack(side="left", fill="both", expand=True)
        # 顶栏
        top = tk.Frame(main, bg=BG_CARD, highlightthickness=1,
                       highlightbackground=BORDER)
        top.pack(fill="x")
        tb = tk.Frame(top, bg=BG_CARD)
        tb.pack(fill="x", padx=16, pady=8)
        self.page_title = tk.Label(tb, text="总览", bg=BG_CARD, fg=TEXT,
                                   font=("Microsoft YaHei UI", 13, "bold"))
        self.page_title.pack(side="left")
        self.page_sub = tk.Label(tb, text="数据概览与模型状态", bg=BG_CARD,
                                 fg=MUTED, font=FONT_SM)
        self.page_sub.pack(side="left", padx=(8, 0), pady=(3, 0))
        self.entry_btn = RoundedButton(tb, "✎ 录入开奖", command=self._manual_entry,
                                       kind="outline", font=FONT_SM)
        self.entry_btn.pack(side="right", padx=(6, 0))
        self.run_btn = RoundedButton(tb, "▶ 开始分析", command=self.start,
                                     kind="primary")
        self.run_btn.pack(side="right", padx=(6, 0))
        self.prog = ttk.Progressbar(tb, mode="indeterminate", length=110)
        self.prog.pack(side="right", padx=10)
        ttk.Checkbutton(tb, text="强制刷新", variable=self.refresh_var).pack(side="right")
        tk.Label(tb, text="回测期数", bg=BG_CARD, fg=TEXT2,
                 font=FONT_SM).pack(side="right", padx=(10, 4))
        ttk.Spinbox(tb, from_=5, to=100, textvariable=self.bt_var,
                    width=4, font=FONT_SM).pack(side="right")

        # 即将开奖横幅(红渐变 + 右侧白色玻璃舱倒计时)
        self._next_issue = None
        self._next_open_dt = None
        banner = tk.Canvas(main, height=52, bg=BG_MAIN, highlightthickness=0, bd=0)
        banner.pack(fill="x", padx=16, pady=(12, 0))
        self._banner_cv = banner
        self._banner_img = None
        banner.bind("<Configure>",
                    lambda e: self._paint_banner(e.width, e.height))
        banner.create_text(18, 26, text="⏰ 即将开奖", fill="#ffffff",
                           font=("Microsoft YaHei UI", 11, "bold"), anchor="w")
        self._ban_expect = banner.create_text(
            122, 26, text="第 - 期", fill="#ffffff",
            font=("Microsoft YaHei UI", 15, "bold"), anchor="w")
        self._ban_cd = banner.create_text(
            0, 0, text="", fill=PRIMARY, font=("Consolas", 11, "bold"), anchor="e")
        self._set_next_issue(load_cache())
        self._tick_countdown()

        # ---- 内容页栈 ----
        content = tk.Frame(main, bg=BG_MAIN)
        content.pack(fill="both", expand=True)
        self._pages = {}
        for _group, items in NAV_GROUPS:
            for key, _text in items:
                self._pages[key] = tk.Frame(content, bg=BG_MAIN)
        self.areas = {}
        for key in ("overview", "full", "single", "dims", "pools",
                    "zodiac", "zodiac_quad", "zodiac_six"):
            self.areas[key] = ScrollArea(self._pages[key])
        self._scroll_canvases = {a.canvas for a in self.areas.values()}
        self._build_records_page(self._pages["records"])
        self._build_history_page(self._pages["history"])
        # 全局鼠标滚轮: 路由到指针所在的可滚动区
        self.root.bind_all("<MouseWheel>", self._on_wheel)
        self.root.bind_all("<Button-4>", self._on_wheel)
        self.root.bind_all("<Button-5>", self._on_wheel)
        self._render_placeholder()
        self._show_page("overview")
```

注意:旧 `_build_ui` 里的 `self.nb = ttk.Notebook(...)`、`self._records_page`、`self._hist_page`、`_on_tab_changed`、`_hover` 全部删除;`self.run_btn.config(state=...)` 两处(`start()` 与 `_poll`)改为:

```python
# start() 内
        self.run_btn.set_enabled(False)
# _poll() error 分支
        self.run_btn.set_enabled(True)
# _poll() done 分支
        self.run_btn.set_enabled(True)
```

`_poll` done 分支末尾 `self.nb.select(self.areas["overview"].canvas.master)` 改为 `self._show_page("overview")`;`_submit_entry` 末尾 `self.nb.select(self._hist_page)` 改为 `self._show_page("history")`。

- [ ] **Step 3d: 新增 `_show_page` / `_paint_banner`,改造 `_set_next_issue` / `_tick_countdown`**

```python
    # ---------------- 页面切换 ----------------
    def _show_page(self, key):
        """切换内容页: 导航选中态 + 页栈显隐 + 顶栏标题 + 档案页刷新钩子。"""
        if key not in self._pages:
            return
        self._current = key
        for k, it in self._nav_items.items():
            it.set_selected(k == key)
        for k, pg in self._pages.items():
            if k == key:
                pg.pack(fill="both", expand=True)
            else:
                pg.pack_forget()
        title, sub = PAGE_TITLES[key]
        self.page_title.config(text=title)
        self.page_sub.config(text=sub)
        if key == "history":
            self._history_refresh()
        elif key == "records":
            self._records_refresh(reload=True)

    def _paint_banner(self, w, h):
        """重绘横幅底(PIL 红渐变圆角图; 无 PIL 降级矢量纯色), 并重置倒计时锚点。"""
        cv = self._banner_cv
        cv.delete("banbg")
        img = _gradient_pil(max(2, w), max(2, h), PRIMARY_D, PRIMARY_L,
                            radius=12) if _PIL_OK else None
        if img is not None:
            self._banner_img = ImageTk.PhotoImage(img)  # 防 GC
            cv.create_image(0, 0, image=self._banner_img, anchor="nw", tags="banbg")
        else:
            draw_round_rect(cv, 0.5, 0.5, w - 1, h - 1, 12, fill=PRIMARY,
                            tags="banbg")
        cv.tag_lower("banbg")
        cv.coords(self._ban_cd, w - 18, h / 2)
```

`_set_next_issue` 替换为(横幅期号改为 canvas 文本项):

```python
    # ---------------- 即将开奖期次 ----------------
    def _set_next_issue(self, records):
        """根据记录更新"即将开奖"横幅(期号 + 倒计时目标)。"""
        ni = compute_next_issue(records)
        self._next_issue = ni
        self._banner_cv.itemconfigure(
            self._ban_expect, text=f"第 {ni['expect']} 期" if ni else "第 - 期")
        self._next_open_dt = self._parse_dt(ni.get("open_time", "")) if ni else None
```

`_tick_countdown` 替换为(白色玻璃舱 + 红色倒计时文字):

```python
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
                txt = f"距 {mmdd} 开奖  还有 {h:02d}:{m:02d}:{s:02d}"
            else:
                txt = "开奖进行中 · 等待结果"
        else:
            txt = ""
        cv = self._banner_cv
        cv.itemconfigure(self._ban_cd, text=txt)
        cv.delete("cdcap")
        if txt:
            box = cv.bbox(self._ban_cd)
            if box:
                bx1, by1, bx2, by2 = box
                draw_round_rect(cv, bx1 - 10, by1 - 5, bx2 + 10, by2 + 5,
                                (by2 - by1) / 2 + 5, fill="#ffffff", tags="cdcap")
                cv.tag_raise(self._ban_cd, "cdcap")
        self.root.after(1000, self._tick_countdown)
```

- [ ] **Step 3e: `_render_placeholder` 文案微调**(bg/fg 令牌已生效,仅确认卡片调用与新 `_card` 兼容 — 下一任务替换 `_card`;本任务保留旧 `_card`/`_metric_tile` 不动,保证中间态可运行)

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_gui_theme.py -v`
Expected: 全部通过(App 冒烟覆盖 10 页切换)

- [ ] **Step 5: 手动冒烟(请用户在终端运行)**

Run: `! python macau_gui.py`
预期:侧导航/顶栏/红渐变横幅出现,10 项导航可切换,倒计时走动,「开始分析」可跑(此时卡片仍是旧直角样式,属正常中间态)。确认后关闭窗口。

- [ ] **Step 6: Commit**

```bash
git add macau_gui.py tests/test_gui_theme.py
git commit -m "feat(gui): 布局骨架重构 - 侧导航+顶栏+渐变横幅替换 Notebook"
```

### Task 5: 卡片构件升级 + 总览仪表盘重组

**Files:**
- Modify: `macau_gui.py`(`_card`、`_metric_tile`、`_render_overview`;模块级新增 `BAR_GRADS`/`_bar_image`)
- Test: `tests/test_gui_theme.py`(无需新增,App 冒烟覆盖)

- [ ] **Step 1: `_card` 替换为圆角卡片(签名不变,所有调用方零改动)**

```python
    # ---------------- 卡片构件 ----------------
    def _card(self, parent, title="", star=False, accent=PRIMARY, expect="",
              dashed=False):
        """圆角卡片: 白底 + 柔影 + 可选虚线边框; 标题带主色小条; 返回内容 Frame。

        expect 非空时在标题后追加期次标签(如【第2026196期】), 置于"★ 推荐"之前。
        """
        rc = RoundedCard(parent, radius=12, fill=BG_CARD,
                         border=BORDER_G if dashed else BORDER,
                         shadow=not dashed, dash=(4, 3) if dashed else None)
        rc.pack(fill="x", padx=14, pady=(6, 10))
        body = rc.body
        if title:
            hd = tk.Frame(body, bg=BG_CARD)
            hd.pack(fill="x", pady=(0, 8))
            tick = tk.Canvas(hd, width=5, height=16, bg=BG_CARD,
                             highlightthickness=0)
            tick.pack(side="left", padx=(0, 8))
            draw_round_rect(tick, 0.5, 0.5, 4, 15, 2, fill=accent)
            tk.Label(hd, text=title, bg=BG_CARD, fg=TEXT,
                     font=FONT_H).pack(side="left")
            if expect:
                tk.Label(hd, text=f"【第{expect}期】", bg=BG_CARD, fg=TEXT2,
                         font=FONT_SM).pack(side="left", padx=(6, 0), pady=(2, 0))
            if star:
                cap = make_capsule(hd, "★ 推荐", PRIMARY_BG, fg=PRIMARY,
                                   font=("Microsoft YaHei UI", 9, "bold"),
                                   padx=10, pady=2)
                cap.pack(side="left", padx=10)
        return body
```

全局把「科学提示」卡调用改为虚线弱化样式 — 共 6 处(`_render_overview`、`_render_zodiac`、`_render_zodiac_quad`、`_render_zodiac_six`、`_render_record_detail`、`_render_history_detail`):
`self._card(a.inner, "科学提示", accent=MUTED)` → `self._card(a.inner, "🛈 科学提示", accent=MUTED, dashed=True)`

- [ ] **Step 2: `_metric_tile` 替换(圆角磁贴;新增 `small` 模式)**

```python
    def _metric_tile(self, parent, label, value, color=PRIMARY, small=False):
        tile = RoundedCard(parent, radius=10, fill=BG_CARD2, border=BORDER,
                           shadow=False, pad=12)
        tile.pack(side="left", padx=5, fill="x", expand=True)
        tk.Label(tile.body, text=value, bg=BG_CARD2, fg=color,
                 font=("Consolas", 11 if small else 18, "bold")).pack(anchor="w")
        tk.Label(tile.body, text=label, bg=BG_CARD2, fg=MUTED,
                 font=FONT_SM).pack(anchor="w")
        return tile
```

- [ ] **Step 3: 模块级新增数据条渐变助手(放在 `make_capsule` 之后)**

```python
# 模型得分条: 按排名循环 红/靛/琥珀 三系渐变
BAR_GRADS = [(PRIMARY, PRIMARY_L), (INDIGO, "#a5b4fc"), (AMBER, "#fcd34d")]
BAR_SOLID = [PRIMARY, INDIGO, AMBER]
_BAR_IMG_CACHE = {}


def _bar_image(w, h, rank):
    """取(必要则渲染并缓存)一根圆角渐变数据条 PhotoImage; 无 PIL 返回 None。"""
    key = (w, h, rank % len(BAR_GRADS))
    img = _BAR_IMG_CACHE.get(key)
    if img is None:
        c1, c2 = BAR_GRADS[key[2]]
        pil = _gradient_pil(w, h, c1, c2, radius=h // 2)
        if pil is None:
            return None
        img = ImageTk.PhotoImage(pil)
        _BAR_IMG_CACHE[key] = img
    return img
```

- [ ] **Step 4: `_render_overview` 替换为仪表盘版**

```python
    def _render_overview(self, recs, report, last, llm):
        a = self.areas["overview"]
        a.clear()
        # 指标磁贴行(含"最近特码"组合磁贴)
        trow = tk.Frame(a.inner, bg=BG_MAIN)
        trow.pack(fill="x", padx=14, pady=(10, 4))
        self._metric_tile(trow, "历史期数", f"{len(recs)}", TEXT)
        self._metric_tile(trow, "最近期号", f"{last.expect}", TEXT)
        self._special_tile(trow, last)
        self._metric_tile(trow, "数据范围",
                          f"{recs[0].open_time[:10]} ~ {last.open_time[:10]}",
                          TEXT, small=True)
        # 反推理模型
        mcard = self._card(a.inner, "🧠 反推理候选数学模型")
        for rank, (name, score, detail) in enumerate(report["models"]):
            row = tk.Frame(mcard, bg=BG_CARD)
            row.pack(fill="x", pady=3)
            tk.Label(row, text=name, bg=BG_CARD, fg=TEXT2, font=FONT,
                     width=14, anchor="w").pack(side="left")
            bar = tk.Canvas(row, width=120, height=10, bg=BG_CARD,
                            highlightthickness=0)
            bar.pack(side="left", padx=6)
            draw_round_rect(bar, 0.5, 0.5, 119, 9, 5, fill=TRACK)
            fw = max(10, int(120 * score))
            img = _bar_image(fw, 10, rank) if _PIL_OK else None
            if img is not None:
                bar.create_image(0, 0, image=img, anchor="nw")
                bar.image = img  # 防 GC
            else:
                draw_round_rect(bar, 0.5, 0.5, fw - 1, 9, 5,
                                fill=BAR_SOLID[rank % len(BAR_SOLID)])
            tk.Label(row, text=f"{score:.3f}", bg=BG_CARD,
                     fg=BAR_SOLID[rank % len(BAR_SOLID)],
                     font=("Consolas", 10, "bold")).pack(side="left", padx=6)
            tk.Label(row, text=detail, bg=BG_CARD, fg=MUTED,
                     font=FONT_SM).pack(side="left")
        # 大模型
        lcard = self._card(a.inner, "🤖 大模型反推理")
        if llm:
            tk.Label(lcard, text="模型清单: " + (", ".join(llm.inferred_models) or "(未解析)"),
                     bg=BG_CARD, fg=PRIMARY, font=FONT, wraplength=880,
                     justify="left", anchor="w").pack(fill="x", pady=2)
            tk.Label(lcard, text="推理: " + llm.reasoning[:300], bg=BG_CARD,
                     fg=MUTED, font=FONT_SM, wraplength=880, justify="left",
                     anchor="w").pack(fill="x")
        else:
            tk.Label(lcard, text="未启用(编辑 config.ini 填入 api_key)",
                     bg=BG_CARD, fg=MUTED, font=FONT).pack(anchor="w")
        # 免责
        dcard = self._card(a.inner, "🛈 科学提示", accent=MUTED, dashed=True)
        tk.Label(dcard, text=DISCLAIMER, bg=BG_CARD, fg=MUTED, font=FONT_SM,
                 justify="left", anchor="w").pack(fill="x")

    def _special_tile(self, parent, rec):
        """总览组合磁贴: 波色球 + 波色·生肖 + 标签。"""
        tile = RoundedCard(parent, radius=10, fill=BG_CARD2, border=BORDER,
                           shadow=False, pad=12)
        tile.pack(side="left", padx=5, fill="x", expand=True)
        row = tk.Frame(tile.body, bg=BG_CARD2)
        row.pack(anchor="w")
        make_ball(row, f"{rec.special:02d}", "special").pack(side="left")
        col = tk.Frame(row, bg=BG_CARD2)
        col.pack(side="left", padx=8)
        wave = wave_of_number(rec.special) or ""
        zod = rec.zodiacs[6] if len(rec.zodiacs) >= 7 else ""
        tk.Label(col, text=f"{wave} · {zod}", bg=BG_CARD2, fg=TEXT,
                 font=("Microsoft YaHei UI", 11, "bold")).pack(anchor="w")
        tk.Label(col, text="最近特码", bg=BG_CARD2, fg=MUTED,
                 font=FONT_SM).pack(anchor="w")
        return tile
```

(旧 `_render_overview` 里的 `days = ...` 死代码与「数据概览/时间范围」卡片一并删除,由磁贴行替代。)

- [ ] **Step 5: 运行测试 + 手动核对**

Run: `python -m pytest tests/test_gui_theme.py -v` → 全绿
Run: `! python macau_gui.py` → 跑一次「开始分析」,核对总览页:磁贴行/组合磁贴/渐变模型条/虚线免责卡。

- [ ] **Step 6: Commit**

```bash
git add macau_gui.py
git commit -m "feat(gui): 圆角卡片构件 + 总览仪表盘(组合磁贴/渐变模型条)"
```

### Task 6: 预测五页 + 生肖三页换肤(令牌迁移)

**Files:**
- Modify: `macau_gui.py`(`_render_full`、`_render_single`、`_render_dims`、`_render_pools`、`_render_zodiac`、`_render_zodiac_quad`、`_render_zodiac_six`、`_render_record_detail`、`_render_history_detail` 中所有 `fg=PRIMARY_L`)

- [ ] **Step 1: 文字前景全局换主色**

新令牌下 `PRIMARY_L = #fb7185` 在白底上太浅,不再用作文字色。全局替换(仅前景用途,渐变端点/悬停用途不含 `fg=` 前缀,不受影响):

- 查找 `fg=PRIMARY_L` → 全部替换为 `fg=PRIMARY`(约 18 处:预测各页 metric_row、维度表 colors、生肖卡指标、记录详情生肖大字、历史详情 llm_models 与各 metric_row)
- 用 Grep 确认替换后无残留: pattern `fg=PRIMARY_L`, Expected: 0 匹配

- [ ] **Step 2: `_render_dims` 表格加斑马纹**

`_render_dims` 中数据行循环改为(表头不动):

```python
        for i, d in enumerate(dims):
            rbg = "#fafbfc" if i % 2 else BG_CARD
            row = tk.Frame(body, bg=rbg)
            row.pack(fill="x", pady=1)
            sig_short, _sig_full, sig_color = self._sig_verdict(d.lift, d.std_error)
            vals = [d.name, d.value, f"{d.accuracy:.1%}", f"{d.baseline:.1%}",
                    f"{d.lift:+.1%}", f"{d.std_error:.1%}", f"{d.stability:.2f}", sig_short]
            colors = [TEXT, PRIMARY, PRIMARY, MUTED, self._lift_color(d.lift),
                      TEXT, TEXT, sig_color]
            widths = [8, 8, 10, 10, 10, 10, 10, 9]
            for v, c, w in zip(vals, colors, widths):
                tk.Label(row, text=v, bg=rbg, fg=c, font=FONT, width=w,
                         anchor="w").pack(side="left")
```

`_render_history_detail` 的六维度表同样处理(斑马纹 + `fg=PRIMARY`):

```python
            for i, d in enumerate(dims):
                name = d.get("name", "")
                hit = bool(v and v.get("dim_hits", {}).get(name))
                actual_dim = v["dim_actuals"].get(name, "—") if v else "—"
                rbg = "#fafbfc" if i % 2 else BG_CARD
                row = tk.Frame(body, bg=rbg)
                row.pack(fill="x", pady=1)
                vals = [name, str(d.get("value", "")), actual_dim,
                        "✓" if hit else ("✗" if v else "—"),
                        f"{d.get('accuracy', 0):.1%}", f"{d.get('lift', 0):+.1%}"]
                colors = [TEXT, PRIMARY, TEXT,
                          (OK if hit else WARN), PRIMARY,
                          self._lift_color(d.get('lift', 0))]
                widths = [8, 8, 8, 8, 10, 10]
                for val, c, w in zip(vals, colors, widths):
                    tk.Label(row, text=val, bg=rbg, fg=c, font=FONT,
                             width=w, anchor="w").pack(side="left")
```

- [ ] **Step 3: 运行测试 + 手动核对**

Run: `python -m pytest tests/test_gui_theme.py -v` → 全绿
Run: `! python macau_gui.py` → 「开始分析」后逐页核对:三组/五组/维度/集合/三肖/四肖/六肖 的卡片圆角、胶囊显著性、指标数值为深红、维度表斑马纹。

- [ ] **Step 4: Commit**

```bash
git add macau_gui.py
git commit -m "feat(gui): 预测页与生肖页换肤 - 主色指标/斑马纹维度表/胶囊显著性"
```

---

### Task 7: 档案两页 + 录入弹窗换肤

**Files:**
- Modify: `macau_gui.py`(`_build_records_page`、`_records_refresh`、`_build_history_page`、`_manual_entry`)

- [ ] **Step 1: 开奖记录页**

(a) `_build_records_page` 工具栏:「⟳ 刷新」按钮替换为:

```python
        RoundedButton(bar, "⟳ 刷新", kind="ghost", font=FONT_SM,
                      command=lambda: self._records_refresh(reload=True)
                      ).pack(side="left", padx=6)
```

(b) 筛选输入框替换为聚焦环样式:

```python
        ent = tk.Entry(bar, textvariable=self.rec_filter_var, width=14, font=FONT,
                       relief="flat", highlightthickness=1,
                       highlightbackground=BORDER_G, highlightcolor=PRIMARY,
                       bg="#ffffff", fg=TEXT, insertbackground=TEXT)
```

(c) Treeview 斑马纹 — `self.rec_tree` 创建后追加:

```python
        self.rec_tree.tag_configure("even", background="#ffffff")
        self.rec_tree.tag_configure("odd", background="#fafbfc")
```

`_records_refresh` 的 insert 循环改为:

```python
        for i, r in enumerate(rows):
            self.rec_tree.insert("", "end", iid=r.expect,
                                 tags=("odd" if i % 2 else "even",),
                                 values=(r.expect, (r.open_time or "")[:16],
                                         ",".join(f"{n:02d}" for n in r.regular),
                                         f"{r.special:02d}"))
```

- [ ] **Step 2: 历史预测页**

(a) `_build_history_page` 三个按钮替换为:

```python
        b1 = RoundedButton(bar, "⟳ 刷新核对", kind="primary",
                           font=("Microsoft YaHei UI", 9, "bold"),
                           command=self._history_fetch_fresh)
        b1.pack(side="left")
        b2 = RoundedButton(bar, "🗑 删除选中", kind="ghost", font=FONT_SM,
                           command=self._history_delete)
        b2.pack(side="left", padx=6)
        b3 = RoundedButton(bar, "清空历史", kind="danger", font=FONT_SM,
                           command=self._history_clear)
        b3.pack(side="left", padx=6)
```

(b) Treeview 斑马纹 — `self.hist_tree` 的 `tag_configure` 区改为(hit/pending 前景标签保留,背景标签可叠加):

```python
        self.hist_tree.tag_configure("pending", foreground=TEXT2)
        self.hist_tree.tag_configure("hit", foreground=OK)
        self.hist_tree.tag_configure("even", background="#ffffff")
        self.hist_tree.tag_configure("odd", background="#fafbfc")
```

`_history_refresh` 的 insert 循环改为(状态标签 + 斑马纹标签合并):

```python
        for i, r in enumerate(self._hist_runs):
            rid = r.get("run_id", "")
            v = self._hist_verify.get(rid)
            g0 = r.get("groups", [{}])[0] if r.get("groups") else {}
            if v is None:
                basis = r.get("basis_expect", "")
                try:
                    next_expect = str(int(basis) + 1) if basis else ""
                except ValueError:
                    next_expect = ""
                target = f"{next_expect} 待开奖" if next_expect else "待开奖"
                wide_t, zod_t = "—", "—"
                tags = ["pending"]
            else:
                target = f"{v['actual_expect']} 已开"
                wide_t = "✓" if v["wide_hit"] else "✗"
                zod_t = "✓" if v["zodiac_hit"] else "✗"
                tags = ["hit"] if (v["wide_hit"] or v["zodiac_hit"]
                                   or v["zodiac_quad_hit"] or v["zodiac_six_hit"]
                                   or v["special_hit_any"]) else []
            tags.append("odd" if i % 2 else "even")
            self.hist_tree.insert("", "end", iid=rid, tags=tuple(tags),
                                  values=(r.get("created_at", "")[:16],
                                          target,
                                          f"{g0.get('special', 0):02d}",
                                          wide_t, zod_t))
```

- [ ] **Step 3: 录入弹窗换肤**

`_manual_entry` 中:
(a) 三个 `tk.Entry` 统一为聚焦环样式(同 Step 1b 的 Entry 参数,字号分别保持 `FONT` / `("Consolas", 11)` / `FONT`);
(b) 底部两按钮替换为:

```python
        RoundedButton(btns, "取消", kind="ghost", font=FONT,
                      command=win.destroy).pack(side="right", padx=6)
        RoundedButton(btns, "保存并核对", kind="primary",
                      command=lambda: self._submit_entry(win, exp_var, num_var,
                                                         time_var, zmap)
                      ).pack(side="right")
```

(c) 删除弹窗内残留的旧 `tk.Button`/`self._hover` 调用;`win.bind("<Escape>", ...)` 保留。

- [ ] **Step 4: 运行测试 + 手动核对**

Run: `python -m pytest tests/test_gui_theme.py -v` → 全绿
Run: `! python macau_gui.py` → 核对:开奖记录页筛选/刷新/选中详情、历史预测页三按钮与斑马纹、录入弹窗(预览波色球正常、保存流程正常)。

- [ ] **Step 5: Commit**

```bash
git add macau_gui.py
git commit -m "feat(gui): 档案页与录入弹窗换肤 - 表格斑马纹/圆角按钮/输入聚焦环"
```

### Task 8: 全量验证 + Codex 交叉审查 + 收尾

**Files:**
- Modify: `macau_gui.py`(按审查结论修).gitignore(可选)
- 临时文件: `codex_gui_review.diff`(用完即删)

- [ ] **Step 1: 全量测试**

Run: `python -m pytest -q`
Expected: 全部通过(既有套件 + 新增 GUI 测试)

- [ ] **Step 2: 无 PIL 降级验证(请用户在终端运行)**

Run: `! python -c "import macau_gui as m; m._PIL_OK = False; m.main()"`
预期:界面正常出现,横幅/Logo 为纯色圆角,卡片无柔影但圆角正常,球为矢量降级样式,不抛异常。确认后关闭。

- [ ] **Step 3: 人工 UI 核对清单(请用户运行 `! python macau_gui.py` 逐项过)**

1. 侧导航 10 项切换正常,选中(浅红底+红条)/悬停态正确;底部状态卡显示「● 就绪」
2. 顶栏标题/副标题随切页更新;回测期数/强制刷新可用
3. 红渐变横幅期号正确、倒计时每秒走动、白色玻璃舱不抖动
4. 「开始分析」全流程:按钮禁用→进度条→完成→自动跳总览→状态卡变「完成 · 已存档」
5. 总览:磁贴行/组合磁贴(球+波色+生肖)/渐变模型条/LLM 卡/虚线免责卡
6. 三组/五组/维度/集合/三肖/四肖/六肖:圆角卡、胶囊显著性、斑马纹维度表、号码球波色与金环
7. 开奖记录:筛选/刷新/斑马纹/选中详情(大球+生肖+波色三行对齐)
8. 历史预测:三按钮、斑马纹、命中金环、删除/清空确认
9. 录入弹窗:聚焦环输入框、预览球、保存并跳转历史页
10. 窗口最小化/拉伸:卡片随宽度伸缩,圆角不失真

- [ ] **Step 4: Codex 交叉审查(按用户全局规则②,必做)**

生成 diff 并喂给 Codex,只问"哪里会炸":

```bash
cd "D:/software/quantifyWorkspace/liuhequantify"
git diff 77a773a..HEAD -- macau_gui.py tests/test_gui_theme.py > codex_gui_review.diff
cat codex_gui_review.diff | codex exec -c model_reasoning_effort=high "你是资深桌面 GUI 代码审查者。项目背景: Python tkinter 桌面应用「澳门六合彩分析预测-猎手2026」, GUI 集中在 macau_gui.py(约1600行), 预测/数据/存档逻辑在其他模块, 本次改动仅为视觉与布局重设计(浅色现代风 + 左侧自绘导航替换 ttk.Notebook), 不允许改变任何预测逻辑与数据流。故意的设计决策(不要当问题报告): 1) 用 PIL 渲染渐变横幅/Logo/数据条与卡片柔影, 无 PIL 时降级为矢量/纯色; 2) 号码球渲染管线(3D 波色球)保持不变; 3) 页面切换由 ttk.Notebook 改为自绘 _show_page; 4) 状态栏移至侧导航底部卡片; 5) 输入内容为上方完整 diff。只回答一个问题: 哪里会炸? —— 运行时异常(TclError/属性名错误/缓存键错误)、资源泄漏(PhotoImage 未引用被 GC)、状态不同步(导航选中/页面刷新/按钮禁用)、无 PIL 降级路径破绽、Windows 特定问题。按严重度逐条列出, 每条给出 文件:行号 与理由。"
```

- [ ] **Step 5: 逐条裁决 Codex 结论并修复真实问题**

对 Codex 报告的每一条:判定真伪 → 真问题在本步修复(附修复说明)→ 伪/误报记录驳回理由。汇总格式:「它报告了什么 / 我判定真伪 / 理由」。修复后重跑 Step 1。

- [ ] **Step 6: 清理与最终提交**

```bash
rm codex_gui_review.diff
# 可选: 若 .gitignore 尚无 .superpowers/, 追加一行(头脑风暴模型图目录)
git add macau_gui.py tests/test_gui_theme.py
git commit -m "chore(gui): 审查修复与收尾 - GUI 重设计完成"
```

---

## Self-Review 记录(计划作者自查)

**Spec 覆盖核对:**
- §2 设计令牌 → Task 1 ✓(含 MUTED 柔和灰取代此前"弱化文字改黑"的旧决定,以已批准 spec 为准)
- §3 布局骨架(侧导航/顶栏/横幅/页栈/1200×800) → Task 4 ✓
- §4 组件改造:圆角卡片(Task 5)/球投影微调(Task 1c)/磁贴+组合磁贴(Task 5)/胶囊显著性(Task 2)/渐变模型条(Task 5)/圆角按钮(Task 2、7)/表格(Task 4 样式 + Task 7 斑马纹)/录入弹窗(Task 7)/虚线免责卡(Task 5) ✓
- §5 页面映射 10→10 → Task 3 数据模型 + Task 4 页栈 ✓
- §6 实现策略(PIL 预渲染+降级、单文件、逻辑零改动) → 贯穿各任务;降级验证 Task 8 Step 2 ✓
- §7 验证(人工清单 + pytest 全绿) → Task 8 Step 1/3 ✓
- §8 不做项 → 计划中无越界内容 ✓
- 用户全局规则(Codex 审查节点②) → Task 8 Step 4/5 ✓

**Placeholder 扫描:** 无 TBD/TODO;所有代码步骤含完整代码;命令含预期输出 ✓

**类型一致性:** `RoundedCard(radius, fill, border, shadow, pad, dash)` / `.body` / `RoundedButton(text, command, kind, font).set_enabled()` / `make_capsule(parent, text, bg, fg, font, padx, pady)` / `draw_round_rect(cv, x1, y1, x2, y2, r, fill, outline, width, dash, tags)` / `_gradient_pil(w, h, c1, c2, radius)` / `_shadow_pil(w, h, radius)` / `_NavItem(parent, key, text, on_select).set_selected()` / `App._show_page(key)` / `App._pages` / `App._current` — 各任务签名一致 ✓
