# Segmentation Explainer

An interactive, step-by-step view of what happens inside the brain-tumour
segmentation networks.
Instead of "image in, mask out", you scrub through every stage of the forward
pass and see the feature maps, the attention gates, and how the final threshold
turns a probability map into a mask.

```
visualizer/
- extract/                   offline: run the model, capture every stage
  - config.py                the pipeline definition (stages and captions) per model
  - hooks.py                 forward-hook recorder and tensor-to-PNG rendering
  - export_activations.py    main script, writes data/
- data/                      generated assets (git-ignored) plus manifest.json
- web/                       static front-end (no build step)
  - index.html
  - css/style.css
  - js/app.js, js/arch.js
```

## 1. Generate the assets

Run from the repo root, using the project venv:

```bash
venv/bin/python visualizer/extract/export_activations.py            # a few samples
venv/bin/python visualizer/extract/export_activations.py --limit 30 # all deploy test slices
venv/bin/python visualizer/extract/export_activations.py --samples BraTS2021_00260_slice_104
```

Inputs come from `DRUnet_v2_jetson_deploy/test_data/npz/*.npz` (each holds the
`(3, 256, 256)` 2.5D stack and its ground-truth `mask`). Output is about 2 MB
per sample under `visualizer/data/<model>/<sample>/`, plus a top-level
`manifest.json` that the web app reads.

## 2. View it

```bash
cd visualizer
python -m http.server 8777
# open http://localhost:8777/   (the root redirects to /web/)
```

The current view is encoded in the URL hash, so a link such as
`.../web/#model=drunetv2&sample=BraTS2021_00260_slice_104&stage=6` is shareable
and reproduces the exact model, slice and stage.

## Controls

- Play and pause auto-advance through the stages while a pulse and playhead
  travel along the horizontal pipeline and each block lights up as the signal
  reaches it. The speed slider sits next to the button (move it right for
  faster).
- The left and right arrow keys step one stage at a time. The spacebar toggles
  play.
- Click any block, or the orange attention diamonds, to jump to that stage.
- The theme button toggles light and dark, remembered per browser.
- Every stage panel shows the block's input and output tensor shapes, its
  parameter count, and an ordered "what this step computes" list. Feature-map
  images enlarge on click.

## Models

| key | description |
|---|---|
| `drunetv2` | Attention Deep Residual U-Net, has SE channel gains and 3 attention gates |
| `mobilenetv2` | Lightweight MobileNetV2 encoder plus a plain skip U-Net, no SE, no attention |

The architecture map is generated from each model's stage list, so it draws the
correct shape automatically (attention diamonds appear only where the model has
gates).

## What each stage shows (DRUNetv2)

| Stage | Panel |
|---|---|
| 2.5D input | the Z-1 / Z / Z+1 stack as RGB, plus the middle target slice |
| Encoder 1 to 4 | montage of feature-map channels, peak-activation map, and the Squeeze-and-Excitation channel-gain bar chart |
| Dilated bottleneck | the deepest 16x16 by 1024-channel features |
| Attention gates (3) | the psi map (0 to 1) blended over the input, where the model looks, with a live opacity slider |
| Decoder 1 to 4 | feature maps re-drawing the shape after fusing attention-filtered skips |
| Head | probability heat-map, a live threshold slider that recomputes the binary mask, the TP / FP / FN overlay, and the Dice score in the browser |

## Adding another model

1. Add an entry to `MODELS` in `extract/config.py` with an ordered `stages`
   list. Each stage names a hookable submodule path (see
   `model.named_modules()`), its shapes, its operation list, and a caption.
2. Add a `build_<name>()` function in `export_activations.py` and register it in
   `BUILDERS`.
3. Re-run the exporter. The web app picks up any model present in
   `manifest.json` automatically.

The baseline `DRUNet` (`DRUnet/model_drunet.py`) is the obvious third model.

## Ideas not yet built

- Precomputed ablations (attention-off, skip-off) with an A/B toggle on the head.
- Side-by-side compare mode: the same slice through both models at once.
