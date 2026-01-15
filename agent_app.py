import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from groq import Groq
from model import UNET
import base64
import io

# ================= ⚙️ CONFIGURATION =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "my_checkpoint.pth.tar"

st.set_page_config(
    page_title="NeuroScan AI | Diagnostic Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stButton>button {
        width: 100%; border-radius: 5px; height: 3em; background-color: #FF4B4B; color: white;
    }
    .report-box {
        padding: 20px; border-radius: 10px; background-color: #262730; border-left: 5px solid #FF4B4B;
    }
    </style>
    """, unsafe_allow_html=True)

# ================= 🧠 MODEL LOADER =================
@st.cache_resource
def load_unet_model():
    model = UNET(in_channels=1, out_channels=1).to(DEVICE)
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model
    except Exception as e:
        st.error(f"❌ Model Error: {e}")
        return None

unet_model = load_unet_model()

# ================= 🛠️ UTILITIES =================
def preprocess_image(pil_image):
    image_np = np.array(pil_image)
    if len(image_np.shape) == 3:
        image_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        image_gray = image_np

    img_resized = cv2.resize(image_gray, (256, 256))
    img_norm = img_resized / 255.0
    img_tensor = torch.tensor(img_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
    
    return img_tensor, img_resized, image_np

def get_bounding_box(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        return x, y, w, h
    return None

def encode_image_to_base64(pil_image):
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# ================= 🤖 AGENT LOGIC (UPDATED FOR LLAMA 4) =================
def analyze_with_groq(api_key, tumor_image_pil, tumor_size_pixels):
    if not api_key:
        return "⚠️ API Key missing. Please provide Groq API Key."

    try:
        client = Groq(api_key=api_key)
        base64_image = encode_image_to_base64(tumor_image_pil)
        
        prompt = f"""
        Act as an expert Oncologist. Analyze this MRI crop of a brain tumor.
        Tumor Area: {tumor_size_pixels} pixels.

        Provide a structured report:
        1. **Morphology:** Describe texture (heterogeneous/homogeneous) and margins (smooth/irregular).
        2. **Prediction:** Likely Benign or Malignant?
        3. **Severity:** Grade (I-IV) estimate.
        4. **Next Steps:** Recommended clinical action.

        Be concise.
        """
        
        # ✅ Using the NEW Model from your list: Llama 4 Scout
        model_id = "meta-llama/llama-4-scout-17b-16e-instruct" 

        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                ]}
            ],
            model=model_id,
            temperature=0.1,
        )
        return chat_completion.choices[0].message.content

    except Exception as e:
        return f"❌ Groq Error: {str(e)}"

# ================= 🖥️ UI LAYOUT =================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=100)
    st.title("NeuroScan AI")
    st.markdown("---")
    api_key = st.text_input("Groq API Key", type="password")
    st.caption("Powered by **Llama 4 Scout** (Vision)")
    st.markdown("---")

col1, col2 = st.columns([2, 1])
with col1:
    st.title("AI-Powered Tumor Analysis")
    st.markdown("*Advanced Segmentation & Morphological Classification*")

uploaded_file = st.file_uploader("Upload MRI Scan", type=["png", "jpg", "jpeg"])

if uploaded_file:
    pil_img = Image.open(uploaded_file)
    input_tensor, display_img, original_rgb = preprocess_image(pil_img)

    with st.spinner('Agent 1 (Radiologist): Segmenting Tumor...'):
        with torch.no_grad():
            preds = torch.sigmoid(unet_model(input_tensor))
            pred_mask = (preds > 0.5).float().squeeze().cpu().numpy().astype(np.uint8) * 255

    tumor_pixels = np.sum(pred_mask > 0)
    
    if tumor_pixels == 0:
        st.success("✅ Analysis Complete: No Tumor Detected.")
        st.image(display_img, caption="Clean Scan", width=300)
    else:
        c1, c2, c3 = st.columns(3)
        with c1: st.image(display_img, caption="Original MRI", width="stretch")
        with c2: st.image(pred_mask, caption=f"Mask ({tumor_pixels} px)", width="stretch")
        with c3:
            colored_mask = np.zeros((256, 256, 3), dtype=np.uint8)
            colored_mask[:, :, 0] = pred_mask 
            base_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2RGB)
            overlay = cv2.addWeighted(base_img, 0.7, colored_mask, 0.3, 0)
            st.image(overlay, caption="Overlay View", width="stretch")

        st.markdown("---")
        
        bbox = get_bounding_box(pred_mask)
        if bbox:
            x, y, w, h = bbox
            pad = 10
            x, y = max(0, x-pad), max(0, y-pad)
            w, h = min(256-x, w+2*pad), min(256-y, h+2*pad)
            
            cropped_tumor = Image.fromarray(base_img[y:y+h, x:x+w])
            
            st.subheader("📑 Automated Pathology Report")
            if st.button("Generate Diagnostic Report (Agentic Workflow)"):
                with st.spinner('Agent 2 (Consultant): Analyzing Tumor via Llama 4...'):
                    # Call with Llama 4 Scout
                    report = analyze_with_groq(api_key, cropped_tumor, tumor_pixels)
                    
                    st.markdown('<div class="report-box">', unsafe_allow_html=True)
                    st.markdown(report)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    with st.expander("See Agent's Input View"):
                        st.image(cropped_tumor, caption="ROI Crop", width=150)
        else:
            st.warning("Tumor too small for agent analysis.")
else:
    st.info("👈 Upload an MRI scan to begin.")