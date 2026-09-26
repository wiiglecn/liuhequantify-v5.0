# -*- coding: utf-8 -*-
"""真实启动流程: 只靠 after(200) 触发, 捕获任何异常, 检查 overview 实际内容。"""
import sys, os
sys.path.insert(0,"."); os.environ["PYTHONUTF8"]="1"
import tkinter as tk
from PIL import ImageGrab

root = tk.Tk()
from macau_gui import App
app = App(root)  # _build_ui 里 schedule after(200)

checked = {"done": False}
def check():
    root.update()
    print(f"[check] results is None: {app.results is None}")
    # 检查 overview area 内是否有"数据概览"卡片(真实数据) 还是"欢迎使用"(占位)
    a = app.areas["overview"]
    # 遍历 inner 的子控件文本
    texts = []
    def walk(w):
        for c in w.winfo_children():
            t = c.cget("text") if hasattr(c, "cget") else ""
            if t: texts.append(str(t)[:20])
            walk(c)
    walk(a.inner)
    print(f"[check] overview 子控件文本片段: {texts[:8]}")
    x=root.winfo_rootx(); y=root.winfo_rooty(); w=root.winfo_width(); h=root.winfo_height()
    ImageGrab.grab(bbox=(x,y,x+w,y+h)).save("gui_snap2.png")
    root.after(100, root.destroy)

root.after(1500, check)  # after(200) 的 _restore_last_run 应已完成
root.mainloop()
print("done")
