"""图片筛选分类工具 — 独立小软件。

功能：
  1. 自己选择待分类图片文件夹
  2. 可选推理图目录；默认同目录自动配对 *_orig / *_pred
  3. 原图与推理图默认并排查看，再移动分类
  4. 自己选择/新建分类保存目录及子文件夹

运行：python scripts/review_and_classify.py

审阅快捷键:
  1-9  移动到分类   ←/→ 或 A/D  上/下一张
  空格 切换 并排/原图/推理   +/- 缩放   U 撤销   Del 跳过
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageDraw, ImageFont, ImageTk

APP_TITLE = "图片筛选分类工具"
APP_VERSION = "1.3"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
PRED_RE = re.compile(r"^(.+)_pred(?:_(\d+))?(\.[^.]+)$", re.IGNORECASE)
ORIG_RE = re.compile(r"^(.+)_orig(?:_(\d+))?(\.[^.]+)$", re.IGNORECASE)
PLAIN_DEDUP_RE = re.compile(r"^(.+)_(\d+)(\.[^.]+)$", re.IGNORECASE)
VIEW_MODES = ("both", "orig", "pred")
VIEW_LABELS = {"both": "原图+推理并排", "orig": "仅原图", "pred": "仅推理图"}
PANEL_TITLES = ("原图", "推理图")
CONFIG_DIR = Path(os.environ.get("APPDATA", Path.home())) / "ImageClassifyTool"
CONFIG_PATH = CONFIG_DIR / "config.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(APP_TITLE)
    p.add_argument("--source", type=str, default=None)
    p.add_argument("--categories-dir", type=str, default=None)
    return p.parse_args()


def load_config() -> dict:
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_config(data: dict) -> None:
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass


def open_in_explorer(path: Path) -> None:
    os.startfile(str(path))


def list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def load_font(size: int):
    for name in ("msyh.ttc", "msyhbd.ttc", "simhei.ttf", "simsun.ttc", "Deng.ttf"):
        try:
            return ImageFont.truetype(f"C:/Windows/Fonts/{name}", size)
        except OSError:
            continue
    return ImageFont.load_default()


@dataclass
class Sample:
    """One review unit: original + optional inference visualization."""

    stem_label: str
    orig: Path | None = None
    pred: Path | None = None

    @property
    def all_files(self) -> list[Path]:
        return [p for p in (self.orig, self.pred) if p is not None and p.exists()]

    @property
    def has_pair(self) -> bool:
        return (
            self.orig is not None
            and self.pred is not None
            and self.orig.exists()
            and self.pred.exists()
        )

    def exists(self) -> bool:
        return bool(self.all_files)


def parse_pair_name(filename: str) -> tuple[str, str, str, str]:
    """Parse image name -> (base_key, role, ext, dedup_tag).

    role: pred | orig | plain
    dedup_tag: "" or "1", "2" ... from de-duplication rename
    """
    m = PRED_RE.match(filename)
    if m:
        return m.group(1), "pred", m.group(3), m.group(2) or ""
    m = ORIG_RE.match(filename)
    if m:
        return m.group(1), "orig", m.group(3), m.group(2) or ""
    p = Path(filename)
    m = PLAIN_DEDUP_RE.match(filename)
    if m:
        return m.group(1), "plain", m.group(3), m.group(2)
    return p.stem, "plain", p.suffix or ".bmp", ""


def build_samples(folder: Path, pred_folder: Path | None = None) -> list[Sample]:
    """Group _orig/_pred pairs, including de-duplicated names like *_pred_1 + *_orig_1."""
    entries: list[dict] = []
    for p in list_images(folder):
        base, role, ext, tag = parse_pair_name(p.name)
        entries.append({"base": base, "role": role, "ext": ext.lower(), "tag": tag, "path": p})
    if pred_folder and pred_folder.is_dir():
        for p in list_images(pred_folder):
            base, role, ext, tag = parse_pair_name(p.name)
            entries.append({"base": base, "role": role, "ext": ext.lower(), "tag": tag, "path": p})

    by_base: dict[tuple[str, str], list[dict]] = {}
    for item in entries:
        key = (item["base"], item["ext"])
        by_base.setdefault(key, []).append(item)

    samples: list[Sample] = []
    for (base, ext), items in sorted(by_base.items()):
        origs = [i for i in items if i["role"] == "orig"]
        preds = [i for i in items if i["role"] == "pred"]
        plains = [i for i in items if i["role"] == "plain"]
        used: set[int] = set()

        def add_sample(tag: str, orig_path: Path | None, pred_path: Path | None) -> None:
            label = base if not tag else f"{base}_{tag}"
            samples.append(Sample(stem_label=label, orig=orig_path, pred=pred_path))

        orig_by_tag = {i["tag"]: i for i in origs}
        pred_by_tag = {i["tag"]: i for i in preds}
        for tag in sorted(set(orig_by_tag) | set(pred_by_tag), key=lambda t: (t != "", t)):
            o = orig_by_tag.get(tag)
            p = pred_by_tag.get(tag)
            if o and p:
                add_sample(tag, o["path"], p["path"])
                used.add(id(o))
                used.add(id(p))

        leftover_o = [i for i in origs if id(i) not in used]
        leftover_p = [i for i in preds if id(i) not in used]
        for o, p in zip(leftover_o, leftover_p):
            add_sample("", o["path"], p["path"])
            used.add(id(o))
            used.add(id(p))
        for o in leftover_o[len(leftover_p) :]:
            add_sample(o["tag"], o["path"], None)
        for p in leftover_p[len(leftover_o) :]:
            add_sample(p["tag"], None, p["path"])
        for pl in plains:
            add_sample(pl["tag"], pl["path"], None)

    return samples


def unique_target(dst: Path) -> Path:
    if not dst.exists():
        return dst
    base, role, ext, tag = parse_pair_name(dst.name)
    if role in ("pred", "orig"):
        idx = int(tag) if tag else 0
        while True:
            idx += 1
            suffix = f"_{idx}" if idx else ""
            if role == "pred":
                candidate = dst.with_name(f"{base}_pred{suffix}{ext}")
            else:
                candidate = dst.with_name(f"{base}_orig{suffix}{ext}")
            if not candidate.exists():
                return candidate
    stem, suffix = dst.stem, dst.suffix
    idx = 1
    while True:
        candidate = dst.with_name(f"{stem}_{idx}{suffix}")
        if not candidate.exists():
            return candidate
        idx += 1


def move_sample_files(orig: Path | None, pred: Path | None, dest_dir: Path) -> list[tuple[Path, Path]]:
    """Move orig/pred together; apply the same de-dup suffix to both."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    moves: list[tuple[Path, Path]] = []
    pair: list[tuple[str, Path]] = []
    if orig and orig.exists():
        pair.append(("orig", orig))
    if pred and pred.exists():
        pair.append(("pred", pred))
    if not pair:
        return moves

    if len(pair) == 1:
        role, src = pair[0]
        target = unique_target(dest_dir / src.name)
        shutil.move(str(src), str(target))
        moves.append((target, src))
        return moves

    _, first = pair[0]
    base, _, ext, _ = parse_pair_name(first.name)

    def targets_for(dedup: str) -> dict[Path, Path]:
        suffix = f"_{dedup}" if dedup else ""
        out: dict[Path, Path] = {}
        for role, src in pair:
            name = f"{base}_{role}{suffix}{ext}"
            out[src] = dest_dir / name
        return out

    dedup = ""
    targets = targets_for(dedup)
    while any(t.exists() for t in targets.values()):
        dedup = "1" if not dedup else str(int(dedup) + 1)
        targets = targets_for(dedup)

    for src, dst in targets.items():
        shutil.move(str(src), str(dst))
        moves.append((dst, src))
    return moves


def move_files(files: list[Path], dest_dir: Path) -> list[tuple[Path, Path]]:
    """Backward-compatible wrapper for non-paired moves."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    moves: list[tuple[Path, Path]] = []
    for src in files:
        if not src.exists():
            continue
        target = unique_target(dest_dir / src.name)
        shutil.move(str(src), str(target))
        moves.append((target, src))
    return moves


def undo_moves(moves: list[tuple[Path, Path]]) -> None:
    for new_path, old_path in reversed(moves):
        if new_path.exists():
            old_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(new_path), str(old_path))


class ImageClassifyApp:
    """单窗口：路径设置 + 分类管理 + 大图审阅。"""

    def __init__(self, preset_source: str | None = None, preset_cat_dir: str | None = None):
        cfg = load_config()
        self.source: Path | None = None
        self.pred_folder: Path | None = None
        self.categories_dir: Path | None = None
        self.samples: list[Sample] = []
        self.index = 0
        self.view_mode = "both"
        self.zoom = 1.0
        self.undo_stack: list[list[tuple[Path, Path]]] = []
        self.categories: list[str] = []
        self._photo_refs: list[ImageTk.PhotoImage] = []
        self._drag_start: tuple[int, int] | None = None
        self._canvas_offset = [0, 0]

        self.root = tk.Tk()
        self.root.title(f"{APP_TITLE} v{APP_VERSION}")
        self.root.geometry("1360x880")
        self.root.minsize(1024, 680)

        self.source_var = tk.StringVar(value=preset_source or cfg.get("last_source", ""))
        self.pred_dir_var = tk.StringVar(value=cfg.get("last_pred_dir", ""))
        self.cat_dir_var = tk.StringVar(value=preset_cat_dir or cfg.get("last_categories_dir", ""))
        self.new_cat_var = tk.StringVar()
        self.info_var = tk.StringVar(value="请先选择图片文件夹和分类目录")
        self.status_var = tk.StringVar(value="")
        self.src_count_var = tk.StringVar(value="")

        self._build_ui()
        self._bind_keys()
        self._apply_paths_from_vars(silent=True)
        self._refresh_category_panel()

    # ── UI ──────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        # 顶部：路径设置
        setup = ttk.LabelFrame(self.root, text="路径设置", padding=8)
        setup.pack(fill=tk.X, padx=8, pady=(8, 4))

        r1 = ttk.Frame(setup)
        r1.pack(fill=tk.X, pady=2)
        ttk.Label(r1, text="待分类图片：", width=12).pack(side=tk.LEFT)
        ttk.Entry(r1, textvariable=self.source_var, font=("Consolas", 10)).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=4
        )
        ttk.Button(r1, text="选择文件夹", command=self._pick_source, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Label(r1, textvariable=self.src_count_var, width=22).pack(side=tk.LEFT, padx=4)

        r1b = ttk.Frame(setup)
        r1b.pack(fill=tk.X, pady=2)
        ttk.Label(r1b, text="推理图目录：", width=12).pack(side=tk.LEFT)
        ttk.Entry(r1b, textvariable=self.pred_dir_var, font=("Consolas", 10)).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=4
        )
        ttk.Button(r1b, text="选择", command=self._pick_pred_dir, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Label(
            r1b,
            text="可选；留空则自动匹配同目录下 *_pred / *_orig",
            font=("Microsoft YaHei UI", 8),
            foreground="#666",
        ).pack(side=tk.LEFT, padx=4)

        r2 = ttk.Frame(setup)
        r2.pack(fill=tk.X, pady=2)
        ttk.Label(r2, text="分类保存到：", width=12).pack(side=tk.LEFT)
        ttk.Entry(r2, textvariable=self.cat_dir_var, font=("Consolas", 10)).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=4
        )
        ttk.Button(r2, text="选择目录", command=self._pick_cat_dir, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(r2, text="新建目录", command=self._create_cat_root, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(r2, text="打开", command=self._open_cat_root, width=6).pack(side=tk.LEFT, padx=2)

        # 主体：左侧面板 + 右侧预览
        body = ttk.Frame(self.root)
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        left = ttk.Frame(body, width=280)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))
        left.pack_propagate(False)

        # 分类管理
        cat_box = ttk.LabelFrame(left, text="分类子文件夹", padding=6)
        cat_box.pack(fill=tk.X, pady=(0, 6))

        cat_input_row = ttk.Frame(cat_box)
        cat_input_row.pack(fill=tk.X, pady=(0, 4))
        ttk.Entry(cat_input_row, textvariable=self.new_cat_var, font=("Microsoft YaHei UI", 10)).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4)
        )
        ttk.Button(cat_input_row, text="新建", command=self._create_category, width=6).pack(side=tk.LEFT)

        cat_btn_row = ttk.Frame(cat_box)
        cat_btn_row.pack(fill=tk.X, pady=(0, 4))
        ttk.Button(cat_btn_row, text="刷新", command=self._refresh_category_panel, width=8).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(cat_btn_row, text="删除选中", command=self._delete_category, width=10).pack(side=tk.LEFT)
        ttk.Button(cat_btn_row, text="打开选中", command=self._open_selected_category, width=10).pack(side=tk.LEFT, padx=(4, 0))

        cat_list_wrap = ttk.Frame(cat_box)
        cat_list_wrap.pack(fill=tk.X)
        self.cat_listbox = tk.Listbox(cat_list_wrap, height=7, font=("Microsoft YaHei UI", 10), exportselection=False)
        cat_sb = ttk.Scrollbar(cat_list_wrap, orient=tk.VERTICAL, command=self.cat_listbox.yview)
        self.cat_listbox.configure(yscrollcommand=cat_sb.set)
        self.cat_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True)
        cat_sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.cat_listbox.bind("<Double-Button-1>", lambda _e: self._open_selected_category())

        ttk.Label(
            cat_box,
            text="新建后立即在磁盘创建子文件夹；审阅时按 1-9 移动到此分类。",
            font=("Microsoft YaHei UI", 8),
            foreground="#666",
            wraplength=250,
        ).pack(anchor=tk.W, pady=(4, 0))

        # 待审列表
        file_box = ttk.LabelFrame(left, text="待审图片", padding=6)
        file_box.pack(fill=tk.BOTH, expand=True)
        file_wrap = ttk.Frame(file_box)
        file_wrap.pack(fill=tk.BOTH, expand=True)
        self.file_listbox = tk.Listbox(file_wrap, font=("Consolas", 9), exportselection=False)
        file_sb = ttk.Scrollbar(file_wrap, orient=tk.VERTICAL, command=self.file_listbox.yview)
        self.file_listbox.configure(yscrollcommand=file_sb.set)
        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        file_sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.file_listbox.bind("<<ListboxSelect>>", self._on_file_select)

        ttk.Button(left, text="重新加载图片", command=self._reload_images).pack(fill=tk.X, pady=(6, 0))

        # 右侧预览
        right = ttk.Frame(body)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        info_bar = ttk.Frame(right)
        info_bar.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(info_bar, textvariable=self.info_var, font=("Microsoft YaHei UI", 10)).pack(side=tk.LEFT)
        view_box = ttk.Frame(info_bar)
        view_box.pack(side=tk.RIGHT, padx=4)
        ttk.Label(view_box, text="视图:", font=("Microsoft YaHei UI", 9)).pack(side=tk.LEFT, padx=(0, 4))
        self.view_var = tk.StringVar(value="both")
        for mode, label in (("both", "并排"), ("orig", "原图"), ("pred", "推理")):
            ttk.Radiobutton(
                view_box,
                text=label,
                value=mode,
                variable=self.view_var,
                command=self._on_view_changed,
            ).pack(side=tk.LEFT, padx=2)
        ttk.Button(info_bar, text="打开源文件夹", command=self._open_source).pack(side=tk.RIGHT, padx=2)

        canvas_wrap = ttk.Frame(right)
        canvas_wrap.pack(fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(canvas_wrap, bg="#2b2b2b", highlightthickness=0)
        xsb = ttk.Scrollbar(canvas_wrap, orient=tk.HORIZONTAL, command=self.canvas.xview)
        ysb = ttk.Scrollbar(canvas_wrap, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=xsb.set, yscrollcommand=ysb.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        ysb.grid(row=0, column=1, sticky="ns")
        xsb.grid(row=1, column=0, sticky="ew")
        canvas_wrap.rowconfigure(0, weight=1)
        canvas_wrap.columnconfigure(0, weight=1)
        self.canvas.bind("<ButtonPress-1>", self._on_drag_start)
        self.canvas.bind("<B1-Motion>", self._on_drag_move)
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<Configure>", lambda _e: self._render_image())

        # 底部分类按钮 + 快捷键提示
        ttk.Label(
            self.root,
            text="1-9 移动到分类 | ←/→ 上/下张 | 空格 切换并排/原图/推理 | +/- 缩放 | U 撤销 | Del 跳过",
            font=("Microsoft YaHei UI", 9),
        ).pack(fill=tk.X, padx=8, pady=(2, 0))

        self.cat_btn_frame = ttk.LabelFrame(self.root, text="快速分类（移动，非复制）", padding=6)
        self.cat_btn_frame.pack(fill=tk.X, padx=8, pady=4)
        self.cat_buttons_inner = ttk.Frame(self.cat_btn_frame)
        self.cat_buttons_inner.pack(fill=tk.X)

        ttk.Label(self.root, textvariable=self.status_var, font=("Microsoft YaHei UI", 9)).pack(
            fill=tk.X, padx=8, pady=(0, 8)
        )

        self.source_var.trace_add("write", lambda *_: self._on_source_var_changed())
        self.pred_dir_var.trace_add("write", lambda *_: self._on_source_var_changed())
        self.cat_dir_var.trace_add("write", lambda *_: self._on_cat_dir_var_changed())

    def _bind_keys(self) -> None:
        self.root.bind("<Left>", lambda _e: self._prev())
        self.root.bind("<Right>", lambda _e: self._next())
        self.root.bind("a", lambda _e: self._prev())
        self.root.bind("d", lambda _e: self._next())
        self.root.bind("<space>", lambda _e: self._cycle_view())
        self.root.bind("+", lambda _e: self._zoom_by(1.15))
        self.root.bind("=", lambda _e: self._zoom_by(1.15))
        self.root.bind("-", lambda _e: self._zoom_by(1 / 1.15))
        self.root.bind("0", lambda _e: self._fit_zoom())
        self.root.bind("u", lambda _e: self._undo())
        self.root.bind("<Delete>", lambda _e: self._skip_current())
        for i in range(1, 10):
            self.root.bind(str(i), lambda _e, idx=i - 1: self._classify(idx))

    # ── 路径 & 分类管理 ─────────────────────────────────────────────

    def _pick_source(self) -> None:
        initial = self.source_var.get().strip()
        path = filedialog.askdirectory(title="选择待分类图片文件夹", initialdir=initial or None)
        if path:
            self.source_var.set(path)
            if not self.cat_dir_var.get().strip():
                self.cat_dir_var.set(str(Path(path).parent / "分类结果"))

    def _pick_pred_dir(self) -> None:
        initial = self.pred_dir_var.get().strip() or self.source_var.get().strip()
        path = filedialog.askdirectory(title="选择推理图文件夹（可选）", initialdir=initial or None)
        if path:
            self.pred_dir_var.set(path)

    def _on_view_changed(self) -> None:
        self.view_mode = self.view_var.get()
        self._show_current()

    def _pick_cat_dir(self) -> None:
        initial = self.cat_dir_var.get().strip() or self.source_var.get().strip()
        path = filedialog.askdirectory(title="选择分类保存目录", initialdir=initial or None)
        if path:
            self.cat_dir_var.set(path)

    def _create_cat_root(self) -> None:
        initial = self.cat_dir_var.get().strip() or self.source_var.get().strip()
        parent = filedialog.askdirectory(title="选择父目录，将在此新建分类文件夹", initialdir=initial or None)
        if not parent:
            return
        dlg = tk.Toplevel(self.root)
        dlg.title("新建分类目录")
        dlg.geometry("360x120")
        dlg.transient(self.root)
        dlg.grab_set()
        ttk.Label(dlg, text="新文件夹名称：", font=("Microsoft YaHei UI", 10)).pack(pady=(16, 4))
        name_var = tk.StringVar(value="分类结果")
        entry = ttk.Entry(dlg, textvariable=name_var, font=("Microsoft YaHei UI", 10), width=30)
        entry.pack(pady=4)
        entry.focus_set()
        entry.select_range(0, tk.END)

        def ok():
            name = name_var.get().strip()
            if not name:
                messagebox.showwarning("提示", "请输入文件夹名称", parent=dlg)
                return
            new_path = Path(parent) / name
            new_path.mkdir(parents=True, exist_ok=True)
            self.cat_dir_var.set(str(new_path))
            dlg.destroy()

        ttk.Button(dlg, text="确定", command=ok, width=10).pack(pady=8)
        dlg.bind("<Return>", lambda _e: ok())

    def _ensure_cat_dir(self) -> Path | None:
        text = self.cat_dir_var.get().strip()
        if not text:
            messagebox.showwarning("提示", "请先设置「分类保存到」目录")
            return None
        path = Path(text)
        path.mkdir(parents=True, exist_ok=True)
        self.categories_dir = path
        return path

    def _create_category(self) -> None:
        cat_root = self._ensure_cat_dir()
        if cat_root is None:
            return
        name = self.new_cat_var.get().strip()
        if not name:
            messagebox.showwarning("提示", "请输入分类名称")
            return
        folder = cat_root / name
        if folder.exists():
            messagebox.showinfo("提示", f"分类「{name}」已存在")
        else:
            folder.mkdir(parents=True, exist_ok=True)
            self.status_var.set(f"已新建分类文件夹: {name}")
        self.new_cat_var.set("")
        self._refresh_category_panel()
        self._persist_config()

    def _delete_category(self) -> None:
        sel = self.cat_listbox.curselection()
        if not sel or not self.categories_dir:
            return
        name = self.categories[sel[0]]
        folder = self.categories_dir / name
        if not folder.is_dir():
            self._refresh_category_panel()
            return
        if not messagebox.askyesno("删除分类", f"确定删除分类文件夹「{name}」？\n（文件夹内有文件时无法删除）"):
            return
        try:
            if any(folder.iterdir()):
                messagebox.showerror("无法删除", "该分类文件夹内还有文件，请先移走或手动清空。")
                return
            folder.rmdir()
            self.status_var.set(f"已删除分类: {name}")
            self._refresh_category_panel()
        except OSError as e:
            messagebox.showerror("删除失败", str(e))

    def _open_selected_category(self) -> None:
        sel = self.cat_listbox.curselection()
        if not sel or not self.categories_dir:
            return
        folder = self.categories_dir / self.categories[sel[0]]
        folder.mkdir(parents=True, exist_ok=True)
        open_in_explorer(folder)

    def _open_cat_root(self) -> None:
        cat_root = self._ensure_cat_dir()
        if cat_root:
            open_in_explorer(cat_root)

    def _open_source(self) -> None:
        if self.source and self.source.is_dir():
            open_in_explorer(self.source)

    def _refresh_category_panel(self) -> None:
        for w in self.cat_buttons_inner.winfo_children():
            w.destroy()
        self.cat_listbox.delete(0, tk.END)

        cat_text = self.cat_dir_var.get().strip()
        if cat_text:
            root = Path(cat_text)
            if root.is_dir():
                self.categories_dir = root
                self.categories = sorted(p.name for p in root.iterdir() if p.is_dir())
            else:
                self.categories = []
        else:
            self.categories = []

        for i, name in enumerate(self.categories):
            hotkey = f"[{i + 1}] " if i < 9 else "    "
            self.cat_listbox.insert(tk.END, f"{hotkey}{name}")

        for i, name in enumerate(self.categories[:9]):
            ttk.Button(
                self.cat_buttons_inner,
                text=f"[{i + 1}] {name}",
                command=lambda idx=i: self._classify(idx),
            ).pack(side=tk.LEFT, padx=3, pady=2)
        for name in self.categories[9:]:
            ttk.Button(
                self.cat_buttons_inner,
                text=name,
                command=lambda n=name: self._classify_by_name(n),
            ).pack(side=tk.LEFT, padx=3, pady=2)
        if not self.categories:
            ttk.Label(self.cat_buttons_inner, text="请先在左侧新建分类子文件夹").pack(side=tk.LEFT)

    def _persist_config(self) -> None:
        save_config(
            {
                "last_source": self.source_var.get().strip(),
                "last_pred_dir": self.pred_dir_var.get().strip(),
                "last_categories_dir": self.cat_dir_var.get().strip(),
            }
        )

    def _on_source_var_changed(self) -> None:
        self._apply_paths_from_vars(silent=True)

    def _on_cat_dir_var_changed(self) -> None:
        self._refresh_category_panel()
        self._persist_config()

    def _apply_paths_from_vars(self, silent: bool = False) -> None:
        src_text = self.source_var.get().strip()
        pred_text = self.pred_dir_var.get().strip()
        if src_text and Path(src_text).is_dir():
            self.source = Path(src_text)
            self.pred_folder = Path(pred_text) if pred_text and Path(pred_text).is_dir() else None
            self.samples = build_samples(self.source, self.pred_folder)
            self.index = 0
            n = len(self.samples)
            pairs = sum(1 for s in self.samples if s.has_pair)
            self.src_count_var.set(f"共 {n} 组，{pairs} 组有推理图")
            if pairs > 0:
                self.view_mode = "both"
                self.view_var.set("both")
            self._rebuild_file_list()
            self._show_current()
            if not silent:
                self.status_var.set(f"已加载 {n} 组图片")
        else:
            self.source = None
            self.pred_folder = None
            self.samples = []
            self.src_count_var.set("")
            if src_text and not silent:
                self.status_var.set("图片路径无效")

        cat_text = self.cat_dir_var.get().strip()
        if cat_text:
            Path(cat_text).mkdir(parents=True, exist_ok=True)
        self._refresh_category_panel()
        self._persist_config()

    def _reload_images(self) -> None:
        self._apply_paths_from_vars(silent=False)

    # ── 审阅 ────────────────────────────────────────────────────────

    def _rebuild_file_list(self) -> None:
        self.file_listbox.delete(0, tk.END)
        for i, s in enumerate(self.samples):
            mark = "✓" if not s.exists() else " "
            tag = " [原+推理]" if s.has_pair else (" [推理]" if s.pred and s.pred.exists() else "")
            self.file_listbox.insert(tk.END, f"{mark} {s.stem_label}{tag}")
            if i == self.index:
                self.file_listbox.selection_set(i)
                self.file_listbox.see(i)

    def _current_sample(self) -> Sample | None:
        while self.index < len(self.samples):
            s = self.samples[self.index]
            if s.exists():
                return s
            self.index += 1
        return None

    def _show_current(self) -> None:
        self._rebuild_file_list()
        sample = self._current_sample()
        if sample is None:
            self.info_var.set("当前文件夹已全部处理完毕")
            self.canvas.delete("all")
            return
        remaining = sum(1 for s in self.samples if s.exists())
        name = sample.orig.name if sample.orig and sample.orig.exists() else (
            sample.pred.name if sample.pred else sample.stem_label
        )
        pair_hint = " | 原图+推理" if sample.has_pair else ""
        self.info_var.set(
            f"[{self.index + 1}/{len(self.samples)}] 剩余 {remaining}  |  {name}  |  {VIEW_LABELS.get(self.view_mode, self.view_mode)}{pair_hint}"
        )
        self._fit_zoom()

    def _panels_for_view(self, sample: Sample) -> list[tuple[str, Path]]:
        """Return (title, path) panels left-to-right."""
        if self.view_mode == "orig":
            if sample.orig and sample.orig.exists():
                return [("原图", sample.orig)]
            if sample.pred and sample.pred.exists():
                return [("推理图", sample.pred)]
            return []
        if self.view_mode == "pred":
            if sample.pred and sample.pred.exists():
                return [("推理图", sample.pred)]
            if sample.orig and sample.orig.exists():
                return [("原图", sample.orig)]
            return []
        panels: list[tuple[str, Path]] = []
        if sample.orig and sample.orig.exists():
            panels.append(("原图", sample.orig))
        if sample.pred and sample.pred.exists():
            panels.append(("推理图", sample.pred))
        if not panels and sample.orig:
            panels.append(("原图", sample.orig))
        return panels

    def _load_composite(self, sample: Sample) -> Image.Image | None:
        panels = [(title, p) for title, p in self._panels_for_view(sample) if p.exists()]
        if not panels:
            return None
        imgs = [Image.open(p).convert("RGB") for _, p in panels]
        titles = [t for t, _ in panels]
        if len(imgs) == 1:
            return self._draw_title(imgs[0], titles[0])
        h = max(im.height for im in imgs)
        font = load_font(max(18, h // 40))
        title_h = getattr(font, "size", 18) + 16
        scaled: list[Image.Image] = []
        for im, title in zip(imgs, titles):
            if im.height != h:
                w = int(im.width * h / im.height)
                im = im.resize((w, h), Image.Resampling.LANCZOS)
            scaled.append(self._stack_title(im, title, font, title_h))
        total_w = sum(im.width for im in scaled)
        out_h = h + title_h
        out = Image.new("RGB", (total_w, out_h), (40, 40, 40))
        x = 0
        for im in scaled:
            out.paste(im, (x, 0))
            if x > 0:
                draw = ImageDraw.Draw(out)
                draw.line([(x, 0), (x, out_h)], fill=(120, 120, 120), width=2)
            x += im.width
        return out

    def _draw_title(self, img: Image.Image, title: str) -> Image.Image:
        font = load_font(max(18, img.height // 40))
        title_h = getattr(font, "size", 18) + 16
        return self._stack_title(img, title, font, title_h)

    def _stack_title(self, img: Image.Image, title: str, font, title_h: int) -> Image.Image:
        canvas = Image.new("RGB", (img.width, img.height + title_h), (50, 50, 50))
        canvas.paste(img, (0, title_h))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 4), title, fill=(255, 220, 80), font=font)
        return canvas

    def _fit_zoom(self) -> None:
        sample = self._current_sample()
        if sample is None:
            return
        img = self._load_composite(sample)
        if img is None:
            return
        cw = max(self.canvas.winfo_width(), 400)
        ch = max(self.canvas.winfo_height(), 300)
        self.zoom = min(cw / img.width, ch / img.height, 1.0)
        self._canvas_offset = [0, 0]
        self._render_image()

    def _render_image(self) -> None:
        self.canvas.delete("all")
        self._photo_refs.clear()
        sample = self._current_sample()
        if sample is None:
            return
        img = self._load_composite(sample)
        if img is None:
            return
        nw = max(1, int(img.width * self.zoom))
        nh = max(1, int(img.height * self.zoom))
        disp = img.resize((nw, nh), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(disp)
        self._photo_refs.append(photo)
        ox, oy = self._canvas_offset
        self.canvas.create_image(ox, oy, anchor=tk.NW, image=photo)
        self.canvas.configure(
            scrollregion=(0, 0, max(nw, self.canvas.winfo_width()), max(nh, self.canvas.winfo_height()))
        )

    def _zoom_by(self, factor: float) -> None:
        self.zoom = min(max(self.zoom * factor, 0.05), 20.0)
        self._render_image()

    def _on_wheel(self, event) -> None:
        self._zoom_by(1.1 if event.delta > 0 else 1 / 1.1)

    def _on_drag_start(self, event) -> None:
        self._drag_start = (event.x, event.y)

    def _on_drag_move(self, event) -> None:
        if self._drag_start is None:
            return
        self._canvas_offset[0] += event.x - self._drag_start[0]
        self._canvas_offset[1] += event.y - self._drag_start[1]
        self._drag_start = (event.x, event.y)
        self._render_image()

    def _cycle_view(self) -> None:
        i = VIEW_MODES.index(self.view_mode)
        self.view_mode = VIEW_MODES[(i + 1) % len(VIEW_MODES)]
        self.view_var.set(self.view_mode)
        self._show_current()

    def _on_file_select(self, _event=None) -> None:
        sel = self.file_listbox.curselection()
        if sel:
            self.index = sel[0]
            self._show_current()

    def _prev(self) -> None:
        if self.index > 0:
            self.index -= 1
        self._show_current()

    def _next(self) -> None:
        if self.index < len(self.samples) - 1:
            self.index += 1
        self._show_current()

    def _classify_by_name(self, name: str) -> None:
        if name in self.categories:
            self._classify(self.categories.index(name))

    def _classify(self, cat_index: int) -> None:
        if not self.categories_dir or cat_index >= len(self.categories):
            self.status_var.set("请先新建分类子文件夹")
            return
        self._move_to(self.categories_dir / self.categories[cat_index])

    def _move_to(self, dest: Path) -> None:
        sample = self._current_sample()
        if sample is None:
            return
        try:
            moves = move_sample_files(sample.orig, sample.pred, dest)
        except OSError as e:
            messagebox.showerror("移动失败", str(e))
            return
        self.undo_stack.append(moves)
        n = len(moves)
        self.status_var.set(f"已移动 {n} 个文件 → {dest.name}/  （按 U 撤销）")
        self._show_current()

    def _undo(self) -> None:
        if not self.undo_stack:
            self.status_var.set("没有可撤销的操作")
            return
        moves = self.undo_stack.pop()
        try:
            undo_moves(moves)
        except OSError as e:
            messagebox.showerror("撤销失败", str(e))
            return
        self.index = max(0, self.index - 1)
        self.status_var.set("已撤销上一次移动")
        self._show_current()

    def _skip_current(self) -> None:
        if self._current_sample() is None:
            return
        self.index += 1
        self.status_var.set("已跳过（文件未移动）")
        self._show_current()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    args = parse_args()
    app = ImageClassifyApp(preset_source=args.source, preset_cat_dir=args.categories_dir)
    app.run()


if __name__ == "__main__":
    main()
