"""
Multi-window annotator: polygons on ORIGINAL; offline/online preprocessed views for reference.

Windows (toggle with keys):
  MAIN    - original + reference panel(s), annotate on original only
  ROI     - magnified 2x2: original / offline / online / |offline-online| diff
  DIFF    - full-image |offline - online| heatmap (when both refs exist)
  HELP    - shortcut list

Reference sources:
  Online  - apply_defect_preprocess (--mode) on the fly
  Offline - images in --ref-dir (from build_preprocessed_dataset.py), matched by filename stem

Usage:
  # Online reference only
  python scripts/dual_view_annotate.py --dir D:/raw --mode mixed

  # Offline preprocessed folder (faster, fixed pipeline)
  python scripts/dual_view_annotate.py --dir D:/raw --ref-dir D:/raw-mixed --mode mixed

  # Triple panel: original | offline | online
  python scripts/dual_view_annotate.py --dir D:/raw --ref-dir D:/raw-mixed --layout triple

  # Compare only (no annotation)
  python scripts/dual_view_annotate.py --dir D:/raw --ref-dir D:/raw-mixed --compare-only

Keys:
  Left-click on ORIGINAL panel only     add polygon vertex
  Enter / Space                         close polygon
  u / d / s                             undo vertex / delete shape / save
  n / p                                 next / prev image
  +/- / wheel / drag                    zoom / pan (all panels synced)
  v                                     cycle layout: dual-offline, dual-online, triple
  f / g                                 dual mode: show offline / online reference
  t / m / i / h                         toggle ROI / DIFF / single-channel / HELP windows
  c                                     cycle ROI single-channel (RGB / G / B of active ref)
  r                                     reset zoom
  0-9                                   class label
  q / Esc                               quit (auto-save)
"""

from __future__ import annotations

import argparse
import json
import sys
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics.data.defect_preprocess import apply_defect_preprocess

IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}

WIN_MAIN = "1-MAIN annotate on ORIGINAL"
WIN_ROI = "2-ROI magnifier"
WIN_DIFF = "3-OFFLINE vs ONLINE diff"
WIN_HELP = "4-HELP"

BAR_H = 42
GAP = 5
ROI_TILE = 240
ROI_MARGIN = 6


class Layout(str, Enum):
    DUAL_OFFLINE = "dual-offline"
    DUAL_ONLINE = "dual-online"
    TRIPLE = "triple"


def imread_unicode(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def find_offline_ref(ref_dir: Path | None, image_path: Path) -> Path | None:
    if ref_dir is None or not ref_dir.is_dir():
        return None
    stem = image_path.stem
    candidates = [image_path.name, f"{stem}.png", f"{stem}.bmp", f"{stem}.jpg"]
    for name in candidates:
        p = ref_dir / name
        if p.is_file():
            return p
    for ext in IMAGE_SUFFIXES:
        p = ref_dir / f"{stem}{ext}"
        if p.is_file():
            return p
    return None


def load_labelme_shapes(json_path: Path) -> list[dict]:
    if not json_path.exists():
        return []
    try:
        return list(json.loads(json_path.read_text(encoding="utf-8")).get("shapes", []))
    except Exception:
        return []


def save_labelme(json_path: Path, image_path: Path, im: np.ndarray, shapes: list[dict]) -> None:
    h, w = im.shape[:2]
    payload = {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_path.name,
        "imageData": None,
        "imageHeight": h,
        "imageWidth": w,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def shape_to_pts(shape: dict) -> np.ndarray:
    return np.array(shape.get("points", []), dtype=np.float32)


def draw_overlay(
    im: np.ndarray,
    shapes: list[dict],
    current_pts: list[tuple[float, float]],
    color: tuple[int, int, int] = (0, 255, 255),
) -> np.ndarray:
    out = im.copy()
    for s in shapes:
        pts = shape_to_pts(s).astype(np.int32)
        if pts.shape[0] >= 2:
            cv2.polylines(out, [pts], pts.shape[0] >= 3, (0, 200, 90), 2, cv2.LINE_AA)
        label = str(s.get("label", ""))
        if label and pts.shape[0] > 0:
            cv2.putText(out, label, (pts[0, 0] + 4, pts[0, 1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 90), 1, cv2.LINE_AA)
    if current_pts:
        pts = np.array(current_pts, dtype=np.int32)
        for p in pts:
            cv2.circle(out, tuple(p), 4, color, -1, cv2.LINE_AA)
        if len(pts) >= 2:
            cv2.polylines(out, [pts], False, color, 2, cv2.LINE_AA)
    return out


def resize_to_match(im: np.ndarray, target_hw: tuple[int, int], label: str) -> tuple[np.ndarray, str]:
    th, tw = target_hw
    h, w = im.shape[:2]
    if (h, w) == (th, tw):
        return im, label
    resized = cv2.resize(im, (tw, th), interpolation=cv2.INTER_LINEAR)
    return resized, f"{label} (resized {w}x{h}->{tw}x{th})"


def ref_channel_view(im: np.ndarray, ch: str) -> np.ndarray:
    if im is None or ch == "rgb":
        return im
    b, g, r = cv2.split(im)
    m = {"b": b, "g": g, "r": r}.get(ch, g)
    return cv2.merge((m, m, m))


def abs_diff_vis(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = cv2.absdiff(a, b)
    gray = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
    return cv2.applyColorMap(gray, cv2.COLORMAP_JET)


def help_canvas() -> np.ndarray:
    lines = [
        "MAIN: annotate on left ORIGINAL panel only",
        "v=layout  f=offline ref  g=online ref  (dual mode)",
        "t=ROI window  m=DIFF window  i=G-channel  h=this help",
        "Click/LMB=vertex  Enter=close  u=undo  d=del  s=save",
        "n/p=next/prev  +/- / wheel=zoom  drag=pan  r=reset  q=quit",
        "",
        "Offline: --ref-dir (build_preprocessed_dataset.py)",
        "Online:  --mode mixed|point|line (live preprocess)",
        "Triple layout compares both refs side by side.",
    ]
    h, w = 28 * len(lines) + 20, 620
    img = np.full((h, w, 3), 48, dtype=np.uint8)
    for i, line in enumerate(lines):
        cv2.putText(img, line, (12, 22 + i * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 210, 210), 1, cv2.LINE_AA)
    return img


class DualViewSession:
    def __init__(
        self,
        image_dir: Path,
        mode: str,
        labels: list[str],
        start_idx: int,
        ref_dir: Path | None,
        layout: str,
        compare_only: bool,
        roi_size: int,
        show_roi: bool,
        show_diff: bool,
        show_help: bool,
    ) -> None:
        self.image_dir = image_dir
        self.ref_dir = ref_dir
        self.mode = mode
        self.labels = labels or ["defect"]
        self.label_idx = 0
        self.compare_only = compare_only
        self.roi_size = max(120, roi_size)
        self.show_roi = show_roi
        self.show_diff = show_diff
        self.show_help = show_help
        self.ref_channel = "rgb"

        self.files = list_images(image_dir)
        if not self.files:
            raise SystemExit(f"No images in {image_dir}")
        self.idx = max(0, min(start_idx, len(self.files) - 1))

        has_offline = ref_dir is not None
        if layout == "auto":
            self.layout = Layout.TRIPLE if has_offline else Layout.DUAL_ONLINE
        elif layout == "triple":
            self.layout = Layout.TRIPLE
        elif layout == "dual-offline":
            self.layout = Layout.DUAL_OFFLINE
        else:
            self.layout = Layout.DUAL_ONLINE

        self.orig: np.ndarray | None = None
        self.ref_online: np.ndarray | None = None
        self.ref_offline: np.ndarray | None = None
        self.offline_path: Path | None = None
        self.offline_label = "offline N/A"
        self.online_label = f"online ({mode})"
        self.size_warn = ""

        self.shapes: list[dict] = []
        self.current: list[tuple[float, float]] = []
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.mouse_xy: tuple[int, int] | None = None
        self.cursor_img: tuple[float, float] | None = None
        self.drag_last: tuple[int, int] | None = None
        self.panel_w = 0
        self.panel_h = 0
        self.dirty = False
        self._load_image()

    @property
    def image_path(self) -> Path:
        return self.files[self.idx]

    @property
    def json_path(self) -> Path:
        return self.image_path.with_suffix(".json")

    def panel_count(self) -> int:
        if self.layout == Layout.TRIPLE:
            return 3
        return 2

    def panel_titles(self) -> list[str]:
        if self.layout == Layout.TRIPLE:
            return ["ORIGINAL (annotate)", self.offline_label, self.online_label]
        if self.layout == Layout.DUAL_OFFLINE:
            return ["ORIGINAL (annotate)", self.offline_label]
        return ["ORIGINAL (annotate)", self.online_label]

    def panel_images(self) -> list[np.ndarray | None]:
        assert self.orig is not None
        if self.layout == Layout.TRIPLE:
            return [self.orig, self.ref_offline, self.ref_online]
        if self.layout == Layout.DUAL_OFFLINE:
            return [self.orig, self.ref_offline]
        return [self.orig, self.ref_online]

    def active_ref_for_channel(self) -> np.ndarray | None:
        if self.layout == Layout.DUAL_OFFLINE:
            return self.ref_offline
        if self.layout == Layout.DUAL_ONLINE:
            return self.ref_online
        return self.ref_online or self.ref_offline

    def cycle_layout(self) -> None:
        order = [Layout.DUAL_ONLINE, Layout.DUAL_OFFLINE, Layout.TRIPLE]
        if self.ref_dir is None:
            order = [Layout.DUAL_ONLINE]
        i = order.index(self.layout) if self.layout in order else 0
        self.layout = order[(i + 1) % len(order)]
        if self.ref_dir is None and self.layout != Layout.DUAL_ONLINE:
            self.layout = Layout.DUAL_ONLINE
        print(f"layout -> {self.layout.value}")

    def _load_image(self) -> None:
        path = self.image_path
        im = imread_unicode(path)
        if im is None:
            raise RuntimeError(f"Cannot read {path}")
        self.orig = im
        h, w = im.shape[:2]
        self.panel_w, self.panel_h = w, h
        self.zoom = 1.0
        self.pan_x = self.pan_y = 0.0
        self.size_warn = ""

        self.ref_online = apply_defect_preprocess(im, mode=self.mode)
        off_path = find_offline_ref(self.ref_dir, path)
        self.offline_path = off_path
        if off_path:
            off_im = imread_unicode(off_path)
            if off_im is not None:
                self.ref_offline, self.offline_label = resize_to_match(off_im, (h, w), f"offline ({off_path.name})")
            else:
                self.ref_offline = None
                self.offline_label = "offline (read fail)"
        else:
            self.ref_offline = None
            self.offline_label = "offline (not found)"

        if self.ref_offline is None and self.layout == Layout.DUAL_OFFLINE:
            self.layout = Layout.DUAL_ONLINE
            print(f"no offline for {path.name}, switch to dual-online")

        if not self.compare_only:
            self.shapes = load_labelme_shapes(self.json_path)
        else:
            self.shapes = load_labelme_shapes(self.json_path)
        self.current = []
        self.dirty = False
        print(f"[{self.idx + 1}/{len(self.files)}] {path.name}  layout={self.layout.value}")

    def save(self) -> None:
        if self.compare_only or self.orig is None:
            return
        save_labelme(self.json_path, self.image_path, self.orig, self.shapes)
        self.dirty = False
        print(f"saved {self.json_path}")

    def _view_roi(self) -> tuple[int, int, int, int]:
        z = self.zoom
        vw = max(1, int(round(self.panel_w / z)))
        vh = max(1, int(round(self.panel_h / z)))
        cx = self.panel_w * 0.5 - self.pan_x / z
        cy = self.panel_h * 0.5 - self.pan_y / z
        x0 = int(max(0, min(self.panel_w - vw, round(cx - vw * 0.5))))
        y0 = int(max(0, min(self.panel_h - vh, round(cy - vh * 0.5))))
        x1 = min(self.panel_w, x0 + vw)
        y1 = min(self.panel_h, y0 + vh)
        return x0, y0, x1, y1

    def _render_one_panel(self, im: np.ndarray | None, overlay: bool, placeholder: str) -> np.ndarray:
        tile = np.full((self.panel_h, self.panel_w, 3), 55, dtype=np.uint8)
        if im is None:
            cv2.putText(tile, placeholder, (20, self.panel_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (140, 140, 200), 2, cv2.LINE_AA)
            return tile
        x0, y0, x1, y1 = self._view_roi()
        src = draw_overlay(im, self.shapes, self.current) if overlay else im
        crop = src[y0:y1, x0:x1]
        return cv2.resize(crop, (self.panel_w, self.panel_h), interpolation=cv2.INTER_LINEAR)

    def _canvas_to_image(self, panel_idx: int, cx: int, cy: int) -> tuple[float, float] | None:
        if panel_idx != 0 or cy < 0:
            return None
        x0, y0, x1, y1 = self._view_roi()
        vw, vh = x1 - x0, y1 - y0
        if vw <= 0 or vh <= 0:
            return None
        ix = x0 + (cx / self.panel_w) * vw
        iy = y0 + (cy / self.panel_h) * vh
        return float(np.clip(ix, 0, self.panel_w - 1)), float(np.clip(iy, 0, self.panel_h - 1))

    def _panel_origin_x(self, panel_idx: int) -> int:
        return panel_idx * (self.panel_w + GAP)

    def _display_to_panel(self, x: int, y: int) -> tuple[int, int, int] | None:
        if y < BAR_H:
            return None
        py = y - BAR_H
        for i in range(self.panel_count()):
            x0 = self._panel_origin_x(i)
            if x0 <= x < x0 + self.panel_w:
                return i, x - x0, py
        return None

    def render_main(self) -> np.ndarray:
        n = self.panel_count()
        w = n * self.panel_w + (n - 1) * GAP
        h = self.panel_h + BAR_H
        canvas = np.full((h, w, 3), 40, dtype=np.uint8)
        titles = self.panel_titles()
        images = self.panel_images()
        for i, (title, im) in enumerate(zip(titles, images)):
            ph = "missing" if im is None else ""
            tile = self._render_one_panel(im, overlay=(i == 0), placeholder=ph)
            x0 = self._panel_origin_x(i)
            canvas[BAR_H : BAR_H + self.panel_h, x0 : x0 + self.panel_w] = tile
            color = (80, 255, 180) if i == 0 else (180, 200, 255)
            cv2.putText(canvas, title, (x0 + 8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
            if i > 0:
                cv2.line(canvas, (x0, 0), (x0, h), (65, 65, 65), 1)

        status = f"[{self.idx + 1}/{len(self.files)}] {self.image_path.name}"
        if self.compare_only:
            status += "  [COMPARE ONLY]"
        else:
            status += f"  label={self.labels[self.label_idx]}"
        cv2.putText(canvas, status, (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (190, 190, 190), 1, cv2.LINE_AA)

        if self.mouse_xy and self.cursor_img:
            mx, my = self.mouse_xy
            cv2.line(canvas, (mx, BAR_H), (mx, h), (80, 80, 80), 1)
            cv2.line(canvas, (0, my), (w, my), (80, 80, 80), 1)
            ix, iy = self.cursor_img
            x0, y0, x1, y1 = self._view_roi()
            vw, vh = max(1, x1 - x0), max(1, y1 - y0)
            for i in range(n):
                px = int(round((ix - x0) / vw * self.panel_w))
                py = int(round((iy - y0) / vh * self.panel_h))
                ox = self._panel_origin_x(i) + px
                cv2.drawMarker(canvas, (ox, BAR_H + py), (0, 220, 255), cv2.MARKER_CROSS, 14, 1, cv2.LINE_AA)
        return canvas

    def _crop_at(self, im: np.ndarray | None, ix: float, iy: float, half: int) -> np.ndarray:
        blank = np.full((ROI_TILE, ROI_TILE, 3), 50, dtype=np.uint8)
        if im is None or self.orig is None:
            return blank
        h, w = im.shape[:2]
        x0 = int(max(0, min(w - 1, ix - half)))
        y0 = int(max(0, min(h - 1, iy - half)))
        x1 = int(min(w, ix + half))
        y1 = int(min(h, iy + half))
        if x1 <= x0 or y1 <= y0:
            return blank
        crop = im[y0:y1, x0:x1]
        if self.shapes or self.current:
            crop = draw_overlay(im, self.shapes, self.current)[y0:y1, x0:x1]
        return cv2.resize(crop, (ROI_TILE, ROI_TILE), interpolation=cv2.INTER_LINEAR)

    def render_roi(self) -> np.ndarray | None:
        if self.cursor_img is None or self.orig is None:
            return None
        ix, iy = self.cursor_img
        half = int(self.roi_size * 0.5 / max(self.zoom, 0.5))
        half = max(20, min(half, 400))
        tiles = [
            ("original", self._crop_at(self.orig, ix, iy, half)),
            ("offline", self._crop_at(self.ref_offline, ix, iy, half)),
            ("online", self._crop_at(ref_channel_view(self.active_ref_for_channel(), self.ref_channel), ix, iy, half)),
        ]
        if self.ref_offline is not None and self.ref_online is not None:
            h, w = self.orig.shape[:2]
            o, _ = resize_to_match(self.ref_offline, (h, w), "")
            on = self.ref_online
            x0 = int(max(0, ix - half))
            y0 = int(max(0, iy - half))
            x1 = int(min(w, ix + half))
            y1 = int(min(h, iy + half))
            diff = abs_diff_vis(o[y0:y1, x0:x1], on[y0:y1, x0:x1])
            tiles.append(("|off-on|", cv2.resize(diff, (ROI_TILE, ROI_TILE))))
        else:
            tiles.append(("diff N/A", np.full((ROI_TILE, ROI_TILE, 3), 50, dtype=np.uint8)))

        cols = 2
        rows = (len(tiles) + 1) // 2
        th = ROI_TILE * rows + ROI_MARGIN * (rows + 1) + 30
        tw = ROI_TILE * cols + ROI_MARGIN * (cols + 1)
        canvas = np.full((th, tw, 3), 35, dtype=np.uint8)
        for k, (name, tile) in enumerate(tiles):
            r, c = divmod(k, cols)
            y0 = 30 + ROI_MARGIN + r * (ROI_TILE + ROI_MARGIN)
            x0 = ROI_MARGIN + c * (ROI_TILE + ROI_MARGIN)
            canvas[y0 : y0 + ROI_TILE, x0 : x0 + ROI_TILE] = tile
            cv2.putText(canvas, name, (x0, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        return canvas

    def render_diff_full(self) -> np.ndarray | None:
        if self.ref_offline is None or self.ref_online is None or self.orig is None:
            return None
        h, w = self.orig.shape[:2]
        off, _ = resize_to_match(self.ref_offline, (h, w), "")
        vis = abs_diff_vis(off, self.ref_online)
        if self.shapes:
            vis = draw_overlay(vis, self.shapes, self.current)
        bar = 36
        out = np.full((h + bar, w, 3), 40, dtype=np.uint8)
        out[bar:] = vis
        cv2.putText(out, "|offline - online|  (verify offline dataset)", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 255), 2, cv2.LINE_AA)
        return out

    def close_polygon(self) -> None:
        if self.compare_only or len(self.current) < 3:
            self.current = []
            return
        self.shapes.append(
            {
                "label": self.labels[self.label_idx],
                "points": [[float(x), float(y)] for x, y in self.current],
                "group_id": None,
                "description": "",
                "shape_type": "polygon",
                "flags": {},
            }
        )
        self.current = []
        self.dirty = True

    def add_point(self, ix: float, iy: float) -> None:
        if self.compare_only:
            return
        self.current.append((ix, iy))
        self.dirty = True

    def undo_point(self) -> None:
        if self.current:
            self.current.pop()
            self.dirty = True

    def delete_last_shape(self) -> None:
        if self.shapes:
            self.shapes.pop()
            self.dirty = True

    def next_image(self, delta: int) -> None:
        if self.dirty:
            self.save()
        self.idx = (self.idx + delta) % len(self.files)
        self._load_image()

    def zoom_at(self, factor: float, cx: int, cy: int) -> None:
        old = self.zoom
        self.zoom = float(np.clip(self.zoom * factor, 0.2, 20.0))
        if abs(self.zoom - old) < 1e-6:
            return
        hit = self._display_to_panel(cx, cy)
        if not hit:
            return
        panel_idx, px, py = hit
        coord = self._canvas_to_image(panel_idx, px, py)
        if not coord:
            return
        ix, iy = coord
        x0, y0, x1, y1 = self._view_roi()
        vw_new = self.panel_w / self.zoom
        vh_new = self.panel_h / self.zoom
        rel_x = (ix - x0) / max(1, x1 - x0)
        rel_y = (iy - y0) / max(1, y1 - y0)
        cx_new = ix - rel_x * vw_new
        cy_new = iy - rel_y * vh_new
        self.pan_x = self.panel_w * 0.5 - (cx_new + vw_new * 0.5) * self.zoom
        self.pan_y = self.panel_h * 0.5 - (cy_new + vh_new * 0.5) * self.zoom


def run_session(session: DualViewSession) -> None:
    def on_mouse_main(event, x, y, flags, _ud) -> None:
        session.mouse_xy = (x, y)
        hit = session._display_to_panel(x, y)
        if hit:
            panel_idx, px, py = hit
            c = session._canvas_to_image(panel_idx, px, py)
            session.cursor_img = c
        if event == cv2.EVENT_LBUTTONDOWN and hit and hit[0] == 0 and not session.compare_only:
            c = session._canvas_to_image(0, hit[1], hit[2])
            if c:
                session.add_point(*c)
        elif event == cv2.EVENT_MOUSEWHEEL:
            session.zoom_at(1.15 if flags > 0 else 1 / 1.15, x, y)
        elif event in (cv2.EVENT_RBUTTONDOWN, cv2.EVENT_MBUTTONDOWN) or (
            event == cv2.EVENT_LBUTTONDOWN and hit and hit[0] != 0
        ):
            session.drag_last = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and session.drag_last and (
            flags & cv2.EVENT_FLAG_LBUTTON or flags & cv2.EVENT_FLAG_RBUTTON or flags & cv2.EVENT_FLAG_MBUTTON
        ):
            dx = x - session.drag_last[0]
            dy = y - session.drag_last[1]
            session.pan_x += dx
            session.pan_y += dy
            session.drag_last = (x, y)
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP, cv2.EVENT_MBUTTONUP):
            session.drag_last = None

    cv2.namedWindow(WIN_MAIN, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN_MAIN, on_mouse_main)
    if session.show_roi:
        cv2.namedWindow(WIN_ROI, cv2.WINDOW_AUTOSIZE)
    if session.show_diff:
        cv2.namedWindow(WIN_DIFF, cv2.WINDOW_NORMAL)
    if session.show_help:
        cv2.namedWindow(WIN_HELP, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(WIN_HELP, help_canvas())

    print("Windows: MAIN" + (" ROI" if session.show_roi else "") + (" DIFF" if session.show_diff else "") + (" HELP" if session.show_help else ""))
    print("Keys: v=layout f/g=dual-ref t/m/h/i=windows  (see HELP window)")

    while True:
        cv2.imshow(WIN_MAIN, session.render_main())
        if session.show_roi:
            roi = session.render_roi()
            if roi is not None:
                cv2.imshow(WIN_ROI, roi)
        if session.show_diff:
            diff = session.render_diff_full()
            if diff is not None:
                cv2.imshow(WIN_DIFF, diff)
            else:
                blank = np.full((120, 480, 3), 40, dtype=np.uint8)
                cv2.putText(blank, "DIFF: need --ref-dir + online", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 220), 1)
                cv2.imshow(WIN_DIFF, blank)

        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            if session.dirty:
                session.save()
            break
        if key in (ord("n"),):
            session.next_image(1)
        elif key in (ord("p"),):
            session.next_image(-1)
        elif key == ord("s"):
            session.save()
        elif key == ord("u"):
            session.undo_point()
        elif key == ord("d"):
            session.delete_last_shape()
        elif key in (13, 32):
            session.close_polygon()
        elif key in (ord("+"), ord("=")):
            session.zoom_at(1.2, session.panel_w // 2, BAR_H + session.panel_h // 2)
        elif key in (ord("-"), ord("_")):
            session.zoom_at(1 / 1.2, session.panel_w // 2, BAR_H + session.panel_h // 2)
        elif key == ord("r"):
            session.zoom, session.pan_x, session.pan_y = 1.0, 0.0, 0.0
        elif key == ord("v"):
            session.cycle_layout()
        elif key == ord("f") and session.ref_dir:
            session.layout = Layout.DUAL_OFFLINE
            print("layout -> dual-offline")
        elif key == ord("g"):
            session.layout = Layout.DUAL_ONLINE
            print("layout -> dual-online")
        elif key == ord("t"):
            session.show_roi = not session.show_roi
            if session.show_roi:
                cv2.namedWindow(WIN_ROI, cv2.WINDOW_AUTOSIZE)
            else:
                cv2.destroyWindow(WIN_ROI)
        elif key == ord("m"):
            session.show_diff = not session.show_diff
            if session.show_diff:
                cv2.namedWindow(WIN_DIFF, cv2.WINDOW_NORMAL)
            else:
                cv2.destroyWindow(WIN_DIFF)
        elif key == ord("h"):
            session.show_help = not session.show_help
            if session.show_help:
                cv2.namedWindow(WIN_HELP, cv2.WINDOW_AUTOSIZE)
                cv2.imshow(WIN_HELP, help_canvas())
            else:
                cv2.destroyWindow(WIN_HELP)
        elif key == ord("i"):
            session.ref_channel = {"rgb": "g", "g": "b", "b": "r", "r": "rgb"}.get(session.ref_channel, "rgb")
            print(f"ref channel view -> {session.ref_channel}")
        elif ord("0") <= key <= ord("9"):
            li = key - ord("0")
            if li < len(session.labels):
                session.label_idx = li

    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-window annotate on original; offline/online reference")
    p.add_argument("--dir", type=Path, required=True, help="Original images + LabelMe JSON output")
    p.add_argument("--ref-dir", type=Path, default=None, help="Offline preprocessed images (same stems)")
    p.add_argument("--mode", type=str, default="mixed", choices=["none", "point", "line", "mixed"])
    p.add_argument(
        "--layout",
        type=str,
        default="auto",
        choices=["auto", "dual-offline", "dual-online", "triple"],
        help="auto=triple if --ref-dir else dual-online",
    )
    p.add_argument("--labels", type=str, default="defect", help="Comma-separated classes (0-9)")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--compare-only", action="store_true", help="Browse/compare only, no new polygons")
    p.add_argument("--roi-size", type=int, default=180, help="ROI half-size base (pixels at zoom=1)")
    p.add_argument("--no-roi", action="store_true", help="Do not open ROI magnifier window")
    p.add_argument("--no-diff", action="store_true", help="Do not open offline vs online diff window")
    p.add_argument("--no-help", action="store_true", help="Do not open HELP window")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    labels = [x.strip() for x in args.labels.split(",") if x.strip()]
    ref_dir = args.ref_dir.expanduser().resolve() if args.ref_dir else None
    session = DualViewSession(
        image_dir=args.dir.expanduser().resolve(),
        mode=args.mode,
        labels=labels,
        start_idx=args.start,
        ref_dir=ref_dir,
        layout=args.layout,
        compare_only=args.compare_only,
        roi_size=args.roi_size,
        show_roi=not args.no_roi,
        show_diff=not args.no_diff and ref_dir is not None,
        show_help=not args.no_help,
    )
    run_session(session)


if __name__ == "__main__":
    main()