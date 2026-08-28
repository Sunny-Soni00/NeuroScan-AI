"""
Pipeline definition for the segmentation "explainer".

Each model has an ordered list of STAGES. A stage names one hookable submodule
inside the network plus the human-readable story for that step. The exporter
registers a forward hook on `module`, renders the captured tensor, and writes
everything (plus this structure) into data/manifest.json, which the web
front-end reads to draw the pipeline and the per-stage panels.

Per-stage fields
    id, kind      - identity; kind drives layout + which panel is shown
    title         - heading in the UI
    module        - dotted submodule path to hook (None for the synthetic input)
    se_module     - optional: dotted path whose (B,C) output is the SE gate vector
    spatial       - nominal side length of this stage's feature map
    channels      - channel count of this stage's output
    io            - short "in -> out" tensor-shape string
    ops           - ordered list of the actual operations inside this block
    caption       - plain-language "what is happening and why"

kinds: input | encoder | bottleneck | attention | decoder | head
"""

# =========================================================================
#  DRUNetv2 : AttentionDRUNet  (DRUnet_v2/model_drunet_v2.py)
# =========================================================================

_DRB = [
    "Conv 3x3 -> BatchNorm -> ReLU",
    "Conv 3x3 -> BatchNorm",
    "SE attention: global avg-pool -> FC(/16) -> ReLU -> FC -> Sigmoid -> scale each channel",
    "Add residual (1x1 conv if channel count changes) -> ReLU",
]

DRUNETV2_STAGES = [
    dict(
        id="input", kind="input", title="2.5D input stack", module=None,
        spatial=256, channels=3,
        io="3 slices -> 3 x 256 x 256",
        ops=["Take slices Z-1, Z, Z+1", "Stack them as the R / G / B channels",
             "Scale intensities to 0.0 - 1.0"],
        caption="Three neighbouring MRI slices are stacked as channels. The "
                "middle slice is the one we segment; the neighbours give the "
                "network a cheap sense of 3D context without the cost of a full "
                "3D model.",
    ),
    dict(
        id="enc1", kind="encoder", title="Encoder 1", module="downs.0",
        se_module="downs.0.se.fc", spatial=256, channels=64,
        io="3 x 256 x 256  ->  64 x 256 x 256",
        ops=_DRB + ["MaxPool 2x2 feeds the next stage (the skip is saved first)"],
        caption="A Dilated Residual Block at full resolution. Filters react to "
                "edges and local intensity contrast. Squeeze-and-Excitation "
                "then rescales each channel by how useful it looks for this "
                "slice.",
    ),
    dict(
        id="enc2", kind="encoder", title="Encoder 2", module="downs.1",
        se_module="downs.1.se.fc", spatial=128, channels=128,
        io="64 x 128 x 128  ->  128 x 128 x 128",
        ops=_DRB + ["MaxPool 2x2 -> next stage"],
        caption="After one 2x downsample. The receptive field is wider, so "
                "channels start responding to small textures and blobs rather "
                "than single edges.",
    ),
    dict(
        id="enc3", kind="encoder", title="Encoder 3", module="downs.2",
        se_module="downs.2.se.fc", spatial=64, channels=256,
        io="128 x 64 x 64  ->  256 x 64 x 64",
        ops=_DRB + ["MaxPool 2x2 -> next stage"],
        caption="64x64. Feature maps now light up over whole candidate lesion "
                "regions and anatomical structures, not individual pixels.",
    ),
    dict(
        id="enc4", kind="encoder", title="Encoder 4", module="downs.3",
        se_module="downs.3.se.fc", spatial=32, channels=512,
        io="256 x 32 x 32  ->  512 x 32 x 32",
        ops=_DRB + ["MaxPool 2x2 -> bottleneck"],
        caption="32x32, 512 channels. Highly abstract: 'is there tumour-like "
                "tissue in this quadrant', with almost no spatial precision "
                "left.",
    ),
    dict(
        id="bottleneck", kind="bottleneck", title="Dilated bottleneck",
        module="bottleneck", se_module="bottleneck.se.fc",
        spatial=16, channels=1024,
        io="512 x 16 x 16  ->  1024 x 16 x 16",
        ops=["Conv 3x3, dilation 2 -> BatchNorm -> ReLU",
             "Conv 3x3, dilation 2 -> BatchNorm",
             "SE attention -> scale channels",
             "Add residual -> ReLU"],
        caption="16x16 with dilation 2, so every unit sees a large chunk of the "
                "slice without extra parameters. This is the most global, most "
                "semantic view the network ever forms.",
    ),
    dict(
        id="att4", kind="attention", title="Attention gate @ 32", module="attentions.0.psi",
        spatial=32, channels=1,
        io="decoder g:512x32x32 + skip e4:512x32x32  ->  psi:1x32x32",
        ops=["Wg: 1x1 conv on the upsampled decoder signal g",
             "Wl: 1x1 conv on encoder skip e4",
             "ReLU(Wg + Wl) -> 1x1 conv -> BatchNorm -> Sigmoid  = psi (0..1)",
             "Gated skip = e4 * psi  (passed to the decoder instead of raw e4)"],
        caption="The gate compares what the decoder is asking for with what the "
                "encoder skip offers, and outputs a 0..1 map. Bright = 'let "
                "this location through'. The deepest gate usually locks onto "
                "the tumour.",
    ),
    dict(
        id="dec4", kind="decoder", title="Decoder 4", module="ups.1",
        spatial=32, channels=512,
        io="1024 x 16 x 16  (+ gated skip e4)  ->  512 x 32 x 32",
        ops=["ConvTranspose 2x2, stride 2: upsample 16 -> 32, 1024 -> 512 ch",
             "Concatenate the attention-gated skip e4 (512) -> 1024 ch",
             "Dilated Residual Block 1024 -> 512"],
        caption="Transposed-conv upsample of the bottleneck, fused with the "
                "attention-filtered skip, then a residual block. The network "
                "starts re-drawing the shape.",
    ),
    dict(
        id="att3", kind="attention", title="Attention gate @ 64", module="attentions.1.psi",
        spatial=64, channels=1,
        io="g:256x64x64 + skip e3:256x64x64  ->  psi:1x64x64",
        ops=["Wg / Wl 1x1 convs on decoder signal and skip e3",
             "ReLU -> 1x1 conv -> Sigmoid = psi",
             "Gated skip = e3 * psi"],
        caption="Same mechanism one level up. The focus is broader here - it is "
                "keeping edges and surrounding tissue that the finer stages "
                "will need.",
    ),
    dict(
        id="dec3", kind="decoder", title="Decoder 3", module="ups.3",
        spatial=64, channels=256,
        io="512 x 32 x 32  (+ gated skip e3)  ->  256 x 64 x 64",
        ops=["ConvTranspose 2x2: 32 -> 64, 512 -> 256 ch",
             "Concatenate gated skip e3 -> 512 ch",
             "Dilated Residual Block 512 -> 256"],
        caption="64x64. The boundary of the predicted region becomes visible in "
                "the feature maps.",
    ),
    dict(
        id="att2", kind="attention", title="Attention gate @ 128", module="attentions.2.psi",
        spatial=128, channels=1,
        io="g:128x128x128 + skip e2:128x128x128  ->  psi:1x128x128",
        ops=["Wg / Wl 1x1 convs on decoder signal and skip e2",
             "ReLU -> 1x1 conv -> Sigmoid = psi",
             "Gated skip = e2 * psi"],
        caption="Near full resolution. Almost the whole brain is passed "
                "through; the gate is mostly suppressing background outside the "
                "skull.",
    ),
    dict(
        id="dec2", kind="decoder", title="Decoder 2", module="ups.5",
        spatial=128, channels=128,
        io="256 x 64 x 64  (+ gated skip e2)  ->  128 x 128 x 128",
        ops=["ConvTranspose 2x2: 64 -> 128, 256 -> 128 ch",
             "Concatenate gated skip e2 -> 256 ch",
             "Dilated Residual Block 256 -> 128"],
        caption="128x128. Fine detail returns by fusing high-resolution encoder "
                "skips with the semantic decoder stream.",
    ),
    dict(
        id="dec1", kind="decoder", title="Decoder 1", module="ups.7",
        spatial=256, channels=64,
        io="128 x 128 x 128  (+ gated skip e1)  ->  64 x 256 x 256",
        ops=["ConvTranspose 2x2: 128 -> 256, 128 -> 64 ch",
             "Concatenate the (gated) skip e1 -> 128 ch",
             "Dilated Residual Block 128 -> 64"],
        caption="Back to 256x256. The last shared features before the "
                "classification layer.",
    ),
    dict(
        id="head", kind="head", title="Head: probability -> mask",
        module="final", spatial=256, channels=1,
        io="64 x 256 x 256  ->  1 x 256 x 256",
        ops=["Conv 1x1: 64 channels -> 1 logit per pixel",
             "Sigmoid -> tumour probability 0..1",
             "Pick a threshold -> binary mask",
             "Overlap with ground truth -> Dice score"],
        caption="A 1x1 conv produces one logit per pixel. Sigmoid turns it into "
                "a probability. You pick the threshold that turns probability "
                "into a binary mask - drag it and watch Dice move.",
    ),
]

# =========================================================================
#  MobileNetV2-UNet  (MobileNetV2_Seg/model_mobilenetv2.py)
#  ImageNet-pretrained MobileNetV2 encoder + light U-Net decoder.
#  No SE blocks, no attention gates -> a plain skip-connection U-Net.
# =========================================================================

_DEC = lambda skip: [
    "Bilinear upsample x2",
    f"Concatenate encoder skip {skip}",
    "Conv 3x3 -> BatchNorm -> ReLU   (x2)",
]

MOBILENETV2_STAGES = [
    dict(
        id="input", kind="input", title="2.5D input stack", module=None,
        spatial=256, channels=3,
        io="3 slices -> 3 x 256 x 256",
        ops=["Take slices Z-1, Z, Z+1", "Stack as R / G / B", "Scale to 0.0 - 1.0"],
        caption="Same 3-channel input as DRUNetv2. The encoder that follows was "
                "pretrained on ImageNet photos, then fine-tuned on brain MRI.",
    ),
    dict(
        id="enc1", kind="encoder", title="Encoder 1", module="enc1",
        spatial=128, channels=16,
        io="3 x 256 x 256  ->  16 x 128 x 128",
        ops=["Conv 3x3, stride 2 -> BatchNorm -> ReLU6",
             "Inverted residual: expand 1x1 -> depthwise 3x3 -> project 1x1"],
        caption="MobileNetV2 stem + first inverted-residual block. Cheap "
                "depthwise-separable convolutions; reacts to edges and local "
                "contrast at 128x128.",
    ),
    dict(
        id="enc2", kind="encoder", title="Encoder 2", module="enc2",
        spatial=64, channels=24,
        io="16 x 128 x 128  ->  24 x 64 x 64",
        ops=["2 inverted-residual blocks", "First block stride 2 (downsample)"],
        caption="64x64, 24 channels. Small textures and blobs. Far fewer "
                "channels than DRUNetv2 at the same depth - this is the "
                "lightweight trade-off.",
    ),
    dict(
        id="enc3", kind="encoder", title="Encoder 3", module="enc3",
        spatial=32, channels=32,
        io="24 x 64 x 64  ->  32 x 32 x 32",
        ops=["3 inverted-residual blocks", "First block stride 2"],
        caption="32x32. Feature maps begin to concentrate over candidate lesion "
                "regions.",
    ),
    dict(
        id="enc4", kind="encoder", title="Encoder 4", module="enc4",
        spatial=16, channels=96,
        io="32 x 32 x 32  ->  96 x 16 x 16",
        ops=["7 inverted-residual blocks", "First block stride 2"],
        caption="16x16, 96 channels. Abstract 'tumour-like tissue here' signals "
                "with little spatial precision.",
    ),
    dict(
        id="enc5", kind="bottleneck", title="Encoder bottleneck", module="enc5",
        spatial=8, channels=320,
        io="96 x 16 x 16  ->  320 x 8 x 8",
        ops=["4 inverted-residual blocks", "First block stride 2",
             "No dilated block here (unlike DRUNetv2)"],
        caption="8x8, 320 channels - the deepest MobileNetV2 features. The "
                "decoder must recover all spatial detail from the skips alone.",
    ),
    dict(
        id="up5", kind="decoder", title="Decoder 5", module="up5",
        spatial=16, channels=128,
        io="320 x 8 x 8  (+ skip enc4)  ->  128 x 16 x 16",
        ops=_DEC("enc4 (96 ch)") + ["-> 128 ch"],
        caption="Bilinear upsample 8 -> 16, concatenate the whole encoder skip "
                "enc4 (no attention gate), then two 3x3 convs.",
    ),
    dict(
        id="up4", kind="decoder", title="Decoder 4", module="up4",
        spatial=32, channels=64,
        io="128 x 16 x 16  (+ skip enc3)  ->  64 x 32 x 32",
        ops=_DEC("enc3 (32 ch)") + ["-> 64 ch"],
        caption="16 -> 32. Shape starts to reappear in the feature maps.",
    ),
    dict(
        id="up3", kind="decoder", title="Decoder 3", module="up3",
        spatial=64, channels=32,
        io="64 x 32 x 32  (+ skip enc2)  ->  32 x 64 x 64",
        ops=_DEC("enc2 (24 ch)") + ["-> 32 ch"],
        caption="32 -> 64. Predicted boundaries become visible.",
    ),
    dict(
        id="up2", kind="decoder", title="Decoder 2", module="up2",
        spatial=128, channels=16,
        io="32 x 64 x 64  (+ skip enc1)  ->  16 x 128 x 128",
        ops=_DEC("enc1 (16 ch)") + ["-> 16 ch"],
        caption="64 -> 128, fusing the high-resolution enc1 skip.",
    ),
    dict(
        id="up1", kind="decoder", title="Decoder 1", module="up1",
        spatial=256, channels=16,
        io="16 x 128 x 128  ->  16 x 256 x 256",
        ops=["Bilinear upsample x2 (128 -> 256)",
             "Conv 3x3 -> BatchNorm -> ReLU", "No skip at this level"],
        caption="A plain upsample + conv before the classifier - there is no "
                "encoder feature map at 256x256 to skip from.",
    ),
    dict(
        id="head", kind="head", title="Head: probability -> mask",
        module="final", spatial=256, channels=1,
        io="16 x 256 x 256  ->  1 x 256 x 256",
        ops=["Conv 1x1: 16 channels -> 1 logit per pixel",
             "Sigmoid -> tumour probability 0..1",
             "Pick a threshold -> binary mask",
             "Overlap with ground truth -> Dice score"],
        caption="1x1 conv -> one logit per pixel -> sigmoid probability. Drag "
                "the threshold to turn probability into a binary mask and watch "
                "Dice move.",
    ),
]

MODELS = {
    "drunetv2": dict(
        label="DRUNetv2 - Attention Deep Residual U-Net (~33M params)",
        builder="build_drunetv2",
        checkpoint="DRUnet_v2/results/v2_checkpoint.pth.tar",
        in_channels=3,
        stages=DRUNETV2_STAGES,
    ),
    "mobilenetv2": dict(
        label="MobileNetV2-UNet - lightweight, ImageNet-pretrained encoder (~2.6M params)",
        builder="build_mobilenetv2",
        checkpoint="MobileNetV2_Seg/results/mobilenetv2_best.pth.tar",
        in_channels=3,
        stages=MOBILENETV2_STAGES,
    ),
}

# How many individual channels to show in a stage montage.
MONTAGE_CHANNELS = 16
