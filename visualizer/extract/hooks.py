"""
Forward-hook recorder + tensor -> PNG rendering helpers.

Nothing here is model specific: give FeatureRecorder a model and a list of
dotted module paths, run a forward pass, then read `.store[path]` for the
captured output tensor (CPU, detached).
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image
import matplotlib.cm as cm


class FeatureRecorder:
    def __init__(self, model: torch.nn.Module, paths: list[str]):
        self.model = model
        self.store: dict[str, torch.Tensor] = {}
        self._handles = []
        by_name = dict(model.named_modules())
        for p in paths:
            if p is None:
                continue
            if p not in by_name:
                raise KeyError(f"module path not found: {p}")
            self._handles.append(
                by_name[p].register_forward_hook(self._make_hook(p))
            )

    def _make_hook(self, path: str):
        def hook(_module, _inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            self.store[path] = t.detach().float().cpu()
        return hook

    def clear(self):
        self.store.clear()

    def close(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


# --- rendering ------------------------------------------------------------

def _norm(a: np.ndarray, pct: tuple | None = None) -> np.ndarray:
    a = a.astype(np.float32)
    if pct is not None:
        lo, hi = np.percentile(a, pct[0]), np.percentile(a, pct[1])
    else:
        lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-8:
        return np.zeros_like(a)
    return np.clip((a - lo) / (hi - lo), 0, 1)


def save_gray(arr2d: np.ndarray, path: str, size: int | None = None):
    """arr2d in 0..1 -> grayscale PNG."""
    img = Image.fromarray((np.clip(arr2d, 0, 1) * 255).astype(np.uint8))
    if size:
        img = img.resize((size, size), Image.NEAREST)
    img.save(path)


def save_colormap(arr2d: np.ndarray, path: str, size: int = 256,
                  cmap: str = "magma", normalize: bool = True,
                  pct: tuple | None = None, gamma: float = 1.0):
    """arr2d -> colour-mapped PNG (RGB)."""
    a = _norm(arr2d, pct) if normalize else np.clip(arr2d, 0, 1)
    if gamma != 1.0:
        a = a ** gamma
    rgb = (cm.get_cmap(cmap)(a)[..., :3] * 255).astype(np.uint8)
    img = Image.fromarray(rgb)
    if img.size != (size, size):
        img = img.resize((size, size), Image.BILINEAR)
    img.save(path)


def save_overlay(base_gray: np.ndarray, heat2d: np.ndarray, path: str,
                 size: int = 256, cmap: str = "inferno", alpha: float = 0.55):
    """Blend a colour-mapped heat map over a grayscale base slice."""
    base = _norm(base_gray)
    base_img = Image.fromarray((base * 255).astype(np.uint8)).convert("RGB")
    base_img = base_img.resize((size, size), Image.BILINEAR)

    # upscale the heat map to display size first, then colour + weight
    heat = _norm(heat2d)
    heat = np.array(Image.fromarray((heat * 255).astype(np.uint8))
                    .resize((size, size), Image.BILINEAR)).astype(np.float32) / 255.0
    heat_rgb = (cm.get_cmap(cmap)(heat)[..., :3] * 255).astype(np.uint8)
    heat_img = Image.fromarray(heat_rgb)

    # weight the blend by heat intensity so quiet areas stay anatomical
    w_img = Image.fromarray((heat * alpha * 255).astype(np.uint8), mode="L")
    out = Image.composite(heat_img, base_img, w_img)
    out.save(path)


def save_montage(feat_chw: np.ndarray, path: str, n: int = 16, tile: int = 96,
                 pad: int = 3, cmap: str = "viridis"):
    """
    feat_chw: (C, H, W). Pick up to n evenly spaced channels, normalise each
    on its own range, lay them out in a near-square grid.
    """
    c = feat_chw.shape[0]
    idx = np.linspace(0, c - 1, min(n, c)).round().astype(int)
    idx = sorted(set(idx.tolist()))
    cols = int(np.ceil(np.sqrt(len(idx))))
    rows = int(np.ceil(len(idx) / cols))
    mapper = cm.get_cmap(cmap)

    sheet = np.full(
        (rows * tile + (rows + 1) * pad, cols * tile + (cols + 1) * pad, 3),
        18, dtype=np.uint8,
    )
    for k, ch in enumerate(idx):
        r, col = divmod(k, cols)
        t = _norm(feat_chw[ch])
        rgb = (mapper(t)[..., :3] * 255).astype(np.uint8)
        tile_img = np.array(Image.fromarray(rgb).resize((tile, tile), Image.NEAREST))
        y = pad + r * (tile + pad)
        x = pad + col * (tile + pad)
        sheet[y:y + tile, x:x + tile] = tile_img
    Image.fromarray(sheet).save(path)
    return idx


def mean_activation(feat_chw: np.ndarray) -> np.ndarray:
    """Channel-averaged absolute activation."""
    return np.abs(feat_chw).mean(axis=0)


def peak_activation(feat_chw: np.ndarray) -> np.ndarray:
    """Per-pixel strongest channel response - more structure than the mean,
    so pipeline thumbnails look visibly different from slice to slice."""
    return np.abs(feat_chw).max(axis=0)
