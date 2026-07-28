import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
from typing import Dict, Tuple

from DRUnet.model_drunet import DRUNet
from DRUnet_v2.model_drunet_v2 import AttentionDRUNet
from MobileNetV2_Seg.model_mobilenetv2 import MobileNetV2UNet


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 256

MODEL_CONFIG = {
    "DRUNet": {
        "checkpoint": "DRUnet/results/drunet_highcap_best.pth.tar",
        "channels": 1,
        "loader": "state_dict",
    },
    "UNet": {
        "checkpoint": "my_checkpoint.pth.tar",
        "channels": 1,
        "loader": "state_dict",
    },
    "MobileNetV2": {
        "checkpoint": "MobileNetV2_Seg/results/mobilenetv2_best.pth.tar",
        "channels": 3,
        "loader": "raw_or_state_dict",
    },
}


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNET(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: Tuple[int, int, int, int] = (64, 128, 256, 512),
    ):
        super().__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = torch.nn.functional.interpolate(
                    x,
                    size=(skip_connection.shape[2], skip_connection.shape[3]),
                    mode="bilinear",
                    align_corners=False,
                )

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)


def _extract_state_dict(checkpoint_obj, mode: str):
    if mode == "state_dict":
        if isinstance(checkpoint_obj, dict) and "state_dict" in checkpoint_obj:
            return checkpoint_obj["state_dict"]
        return checkpoint_obj

    if isinstance(checkpoint_obj, dict) and "state_dict" in checkpoint_obj:
        return checkpoint_obj["state_dict"]
    return checkpoint_obj


@st.cache_resource
def load_models() -> Dict[str, Dict[str, object]]:
    models = {}
    for name, cfg in MODEL_CONFIG.items():
        model = None
        error = None
        try:
            if name == "DRUNet":
                model = DRUNet(in_channels=1, out_channels=1).to(DEVICE)
            elif name == "UNet":
                model = UNET(in_channels=1, out_channels=1).to(DEVICE)
            elif name == "MobileNetV2":
                model = MobileNetV2UNet(in_channels=3, out_channels=1, pretrained=False).to(DEVICE)

            checkpoint = torch.load(cfg["checkpoint"], map_location=DEVICE)
            state_dict = _extract_state_dict(checkpoint, cfg["loader"])
            model.load_state_dict(state_dict)
            model.eval()

        except Exception as ex:
            error = str(ex)

        models[name] = {
            "model": model,
            "error": error,
            "channels": cfg["channels"],
            "checkpoint": cfg["checkpoint"],
        }
    return models


def preprocess_uploaded_image(pil_image: Image.Image, channels: int):
    image_np = np.array(pil_image.convert("RGB"))
    image_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    img_resized = cv2.resize(image_gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img_norm = img_resized.astype(np.float32) / 255.0

    if channels == 3:
        stacked = np.stack([img_norm, img_norm, img_norm], axis=0)
        tensor = torch.tensor(stacked, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    else:
        tensor = torch.tensor(img_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)

    return tensor, img_resized


def build_overlay(base_gray: np.ndarray, pred_mask: np.ndarray):
    base_rgb = cv2.cvtColor(base_gray.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    color_mask = np.zeros_like(base_rgb)
    color_mask[:, :, 0] = (pred_mask * 255).astype(np.uint8)
    return cv2.addWeighted(base_rgb, 0.7, color_mask, 0.3, 0)


def tumor_report(model_name: str, tumor_present: bool, confidence: float, tumor_pct: float) -> str:
    if not tumor_present:
        return (
            f"Model {model_name} predicts no visible tumor region. "
            f"Confidence: {confidence:.2f}%. "
            "Keep clinical review with full MRI sequences for confirmation."
        )

    severity = "Low" if tumor_pct < 1.0 else "Moderate" if tumor_pct < 5.0 else "High"
    return (
        f"Model {model_name} predicts tumor present with confidence {confidence:.2f}%. "
        f"Estimated tumor coverage is {tumor_pct:.2f}% of the analyzed slice, "
        f"indicating {severity.lower()} burden in this 2D view. "
        "Use this as AI-assisted screening, not a final diagnosis."
    )


def run_prediction(
    model: nn.Module,
    input_tensor: torch.Tensor,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        probs = torch.sigmoid(model(input_tensor))
        pred_mask = (probs > threshold).float().squeeze().cpu().numpy()
    return pred_mask.astype(np.float32), probs.squeeze().cpu().numpy().astype(np.float32)


def compute_confidence(prob_map: np.ndarray, pred_mask: np.ndarray) -> float:
    if np.any(pred_mask > 0.5):
        return float(np.mean(prob_map[pred_mask > 0.5]) * 100.0)
    return float((1.0 - np.mean(prob_map)) * 100.0)


st.set_page_config(
    page_title="Brain Tumor Multi-Model Predictor",
    page_icon="🧠",
    layout="wide",
)

st.title("Brain Tumor Prediction - Multi Model")
st.caption("Upload MRI or screenshot image, run one or multiple models, and view separate tumor predictions with confidence.")

models = load_models()

with st.sidebar:
    st.header("Inference Settings")
    selected_models = st.multiselect(
        "Select One or More Models",
        list(MODEL_CONFIG.keys()),
        default=["DRUNet"],
    )
    threshold = st.slider("Mask Threshold", min_value=0.30, max_value=0.90, value=0.50, step=0.05)

    st.markdown("Model Load Status")
    for model_name in selected_models:
        info = models[model_name]
        if info["error"]:
            st.error(f"{model_name}: load failed")
            st.caption(info["error"])
            st.caption(f"Checkpoint: {info['checkpoint']}")
        else:
            st.success(f"{model_name}: ready")
            st.caption(f"Checkpoint: {info['checkpoint']}")

input_tab1, input_tab2 = st.tabs(["Upload MRI Image", "Upload Screenshot Image"])

uploaded_file = None
screenshot_file = None

with input_tab1:
    uploaded_file = st.file_uploader("Upload MRI image", type=["png", "jpg", "jpeg"])

with input_tab2:
    st.caption("Upload your screenshot image directly (PNG/JPG/JPEG).")
    screenshot_file = st.file_uploader(
        "Upload screenshot image",
        type=["png", "jpg", "jpeg"],
        key="screenshot_uploader",
    )
    if screenshot_file is not None:
        st.image(screenshot_file, caption="Screenshot Preview", use_container_width=True)

source_image = None
if uploaded_file is not None:
    source_image = Image.open(uploaded_file)
elif screenshot_file is not None:
    source_image = Image.open(screenshot_file)

if source_image is not None:
    if not selected_models:
        st.warning("Please select at least one model.")
    else:
        if st.button("Run Prediction", type="primary"):
            st.subheader("Per-Model Results")

            for model_name in selected_models:
                model_info = models[model_name]

                if model_info["error"] or model_info["model"] is None:
                    st.error(f"{model_name}: model not available.")
                    st.caption(model_info["error"])
                    continue

                input_tensor, display_gray = preprocess_uploaded_image(source_image, model_info["channels"])
                pred_mask, prob_map = run_prediction(model_info["model"], input_tensor, threshold)
                overlay = build_overlay(display_gray, pred_mask)

                tumor_pixels = int(np.sum(pred_mask > 0.5))
                total_pixels = int(pred_mask.size)
                tumor_pct = (tumor_pixels / total_pixels) * 100.0 if total_pixels > 0 else 0.0
                confidence = compute_confidence(prob_map, pred_mask)
                tumor_present = tumor_pixels > 100
                decision = "Tumor Detected" if tumor_present else "No Tumor Detected"

                st.markdown("---")
                st.subheader(f"Model: {model_name}")

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.image(display_gray, caption="Input MRI", use_container_width=True)
                with c2:
                    st.image((pred_mask * 255).astype(np.uint8), caption="Predicted Mask", use_container_width=True)
                with c3:
                    st.image(overlay, caption="Overlay", use_container_width=True)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Prediction", decision)
                m2.metric("Confidence", f"{confidence:.2f}%")
                m3.metric("Tumor Pixels", f"{tumor_pixels:,}")
                m4.metric("Tumor Coverage", f"{tumor_pct:.2f}%")

                st.write(tumor_report(model_name, tumor_present, confidence, tumor_pct))

else:
    st.info("Upload MRI image or screenshot image to start prediction.")

st.markdown("---")
st.caption(f"Running on: {DEVICE.upper()}")