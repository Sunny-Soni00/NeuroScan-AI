"""
Run a segmentation model on the deploy test slices, capture every stage of the
forward pass with hooks, and dump PNG/JSON assets for the web explainer.

    python visualizer/extract/export_activations.py                # drunetv2, 6 samples
    python visualizer/extract/export_activations.py --limit 30
    python visualizer/extract/export_activations.py --samples BraTS2021_00025_slice_108

Output goes to visualizer/data/<model>/<sample>/  plus visualizer/data/manifest.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.path.abspath(os.path.join(HERE, "..", "data"))
sys.path.insert(0, REPO)

from visualizer.extract import config as C          # noqa: E402
from visualizer.extract.hooks import (              # noqa: E402
    FeatureRecorder, save_gray, save_colormap, save_overlay, save_montage,
    peak_activation,
)

NPZ_DIR = os.path.join(REPO, "DRUnet_v2_jetson_deploy", "test_data", "npz")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --- model builders ----------------------------------------------------------

def build_drunetv2(checkpoint: str):
    from DRUnet_v2.model_drunet_v2 import AttentionDRUNet
    model = AttentionDRUNet(in_channels=3, out_channels=1).to(DEVICE)
    sd = torch.load(os.path.join(REPO, checkpoint), map_location=DEVICE)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd)
    model.eval()
    return model


def build_mobilenetv2(checkpoint: str):
    from MobileNetV2_Seg.model_mobilenetv2 import MobileNetV2UNet
    model = MobileNetV2UNet(in_channels=3, out_channels=1, pretrained=False).to(DEVICE)
    sd = torch.load(os.path.join(REPO, checkpoint), map_location=DEVICE)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd)
    model.eval()
    return model


BUILDERS = {
    "build_drunetv2": build_drunetv2,
    "build_mobilenetv2": build_mobilenetv2,
}


# --- helpers ---------------------------------------------------------------

def dice(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    p = (pred > 0.5).astype(np.float32)
    g = (gt > 0.5).astype(np.float32)
    return float((2 * (p * g).sum() + eps) / (p.sum() + g.sum() + eps))


def load_sample(npz_path: str):
    d = np.load(npz_path)
    x = d["input"].astype(np.float32)          # (3, 256, 256), already 0..1
    gt = d["mask"].astype(np.float32)          # (256, 256)
    return x, gt


# --- per-model export ------------------------------------------------------

def count_params(model, dotted: str) -> int:
    mods = dict(model.named_modules())
    m = mods.get(dotted)
    if m is None and "." in dotted:                 # attention: count the gate
        m = mods.get(dotted.rsplit(".", 1)[0])
    return int(sum(p.numel() for p in m.parameters())) if m is not None else 0


def export_model(key: str, cfg: dict, npz_files: list[str]) -> dict:
    model = BUILDERS[cfg["builder"]](cfg["checkpoint"])
    stages = cfg["stages"]

    hook_paths = []
    for s in stages:
        if s.get("module"):
            hook_paths.append(s["module"])
        if s.get("se_module"):
            hook_paths.append(s["se_module"])
    rec = FeatureRecorder(model, hook_paths)
    params = {}
    for s in stages:
        mod = s.get("module")
        if not mod:
            params[s["id"]] = 0
        elif s["kind"] == "attention":              # count the whole gate
            params[s["id"]] = count_params(model, mod.rsplit(".", 1)[0])
        else:
            params[s["id"]] = count_params(model, mod)

    sample_entries = []
    for npz_path in npz_files:
        name = os.path.splitext(os.path.basename(npz_path))[0]
        out_dir = os.path.join(DATA, key, name)
        os.makedirs(out_dir, exist_ok=True)
        x, gt = load_sample(npz_path)
        mid = x[1]                                   # middle slice = target

        rec.clear()
        with torch.no_grad():
            logits = model(torch.from_numpy(x).unsqueeze(0).to(DEVICE))
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

        # --- global inputs / outputs
        save_colormap(np.transpose(x, (1, 2, 0)).mean(axis=2),
                      os.path.join(out_dir, "input_rgb.png"), cmap="gray",
                      normalize=True)
        # true RGB stack
        from PIL import Image
        Image.fromarray((np.clip(np.transpose(x, (1, 2, 0)), 0, 1) * 255)
                        .astype(np.uint8)).save(os.path.join(out_dir, "input_stack.png"))
        save_gray(mid, os.path.join(out_dir, "input_mid.png"))
        save_gray(gt, os.path.join(out_dir, "gt_mask.png"))
        save_gray(prob, os.path.join(out_dir, "prob.png"))          # 8-bit prob
        save_colormap(prob, os.path.join(out_dir, "prob_heat.png"),
                      cmap="magma", normalize=False)

        stage_assets = {}
        for s in stages:
            sid = s["id"]
            entry: dict = {}

            if s["kind"] == "input":
                entry = {"stack": "input_stack.png", "mid": "input_mid.png"}

            elif s["kind"] in ("encoder", "bottleneck", "decoder"):
                feat = rec.store[s["module"]][0].numpy()            # (C,H,W)
                shown = save_montage(feat, os.path.join(out_dir, f"{sid}_montage.png"),
                                     n=C.MONTAGE_CHANNELS)
                save_colormap(peak_activation(feat),
                              os.path.join(out_dir, f"{sid}_thumb.png"),
                              cmap="magma", pct=(1, 99), gamma=0.8)
                entry = {"montage": f"{sid}_montage.png",
                         "thumb": f"{sid}_thumb.png",
                         "channels_shown": shown,
                         "shape": list(feat.shape)}
                if s.get("se_module") and s["se_module"] in rec.store:
                    gains = rec.store[s["se_module"]][0].numpy().reshape(-1)
                    entry["se_gains"] = [round(float(v), 4) for v in gains]

            elif s["kind"] == "attention":
                psi = rec.store[s["module"]][0, 0].numpy()          # (H,W) 0..1
                save_overlay(mid, psi, os.path.join(out_dir, f"{sid}_overlay.png"))
                save_colormap(psi, os.path.join(out_dir, f"{sid}_heat.png"),
                              cmap="inferno", normalize=False)
                save_colormap(psi, os.path.join(out_dir, f"{sid}_thumb.png"),
                              cmap="inferno", normalize=False)
                entry = {"overlay": f"{sid}_overlay.png",
                         "heat": f"{sid}_heat.png",
                         "thumb": f"{sid}_thumb.png",
                         "coverage": round(float((psi > 0.5).mean()), 4),
                         "peak": round(float(psi.max()), 4)}

            elif s["kind"] == "head":
                entry = {"prob": "prob.png", "prob_heat": "prob_heat.png",
                         "gt": "gt_mask.png", "thumb": "prob_heat.png"}

            stage_assets[sid] = entry

        sample_entries.append({
            "name": name,
            "dice_at_0.5": round(dice(prob, gt), 4),
            "tumor_pct_gt": round(float((gt > 0.5).mean() * 100), 3),
            "tumor_pct_pred": round(float((prob > 0.5).mean() * 100), 3),
            "stages": stage_assets,
        })
        print(f"  [{key}] {name}  dice@0.5={sample_entries[-1]['dice_at_0.5']:.3f}")

    rec.close()
    return {
        "label": cfg["label"],
        "in_channels": cfg["in_channels"],
        "stages": [
            {k: s[k] for k in ("id", "kind", "title", "caption", "io", "ops")
             if k in s}
            | {"spatial": s.get("spatial"), "channels": s.get("channels"),
               "params": params.get(s["id"], 0)}
            for s in stages
        ],
        "samples": sample_entries,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(C.MODELS.keys()))
    ap.add_argument("--limit", type=int, default=6)
    ap.add_argument("--samples", nargs="+", default=None,
                    help="explicit sample stems (without .npz)")
    args = ap.parse_args()

    all_npz = sorted(glob.glob(os.path.join(NPZ_DIR, "*.npz")))
    if args.samples:
        want = set(args.samples)
        npz_files = [p for p in all_npz
                     if os.path.splitext(os.path.basename(p))[0] in want]
    else:
        npz_files = all_npz[: args.limit]
    if not npz_files:
        raise SystemExit(f"no npz samples found under {NPZ_DIR}")

    os.makedirs(DATA, exist_ok=True)
    manifest = {"device": DEVICE, "models": {}}
    for key in args.models:
        print(f"== exporting {key} ({len(npz_files)} samples) ==")
        manifest["models"][key] = export_model(key, C.MODELS[key], npz_files)

    with open(os.path.join(DATA, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nwrote {os.path.join(DATA, 'manifest.json')}")
    print(f"assets under {DATA}/")


if __name__ == "__main__":
    main()
