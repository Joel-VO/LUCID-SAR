import io
import numpy as np
from PIL import Image
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18
from fastai.vision.learner import create_body
from fastai.vision.models.unet import DynamicUnet
from skimage.color import lab2rgb
import streamlit as st

# ─────────────────────────────────────────────
# Hardcoded model paths  ← edit these
# ─────────────────────────────────────────────
DESPECKLE_WEIGHTS = "SAR/models/denoiser/idcnn_inception_reduced.pth"
COLORIZER_WEIGHTS = "SAR/models/Colorizer/generator_final.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────
# Page config + custom CSS
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SAR Processor",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow:wght@300;400;600&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background-color: #0b0e14 !important;
    color: #c8d6e5 !important;
    font-family: 'Barlow', sans-serif !important;
}
[data-testid="stSidebar"] {
    background-color: #0d1117 !important;
    border-right: 1px solid #1f2d3d !important;
}
.sar-header {
    font-family: 'Share Tech Mono', monospace;
    font-size: 2rem;
    letter-spacing: 0.25em;
    color: #39ff9a;
    text-transform: uppercase;
    border-bottom: 1px solid #1f3a2a;
    padding-bottom: 0.5rem;
    margin-bottom: 0.25rem;
}
.sar-sub {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    color: #4a6a5a;
    letter-spacing: 0.15em;
    margin-bottom: 2rem;
}
.badge {
    display: inline-block;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    padding: 3px 10px;
    border-radius: 2px;
    margin-bottom: 1.5rem;
}
.badge-gpu { background: #0a2e1a; color: #39ff9a; border: 1px solid #39ff9a44; }
.badge-cpu { background: #2e1a0a; color: #ffb347; border: 1px solid #ffb34744; }
.section-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem;
    color: #4a6a5a;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.img-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.7rem;
    color: #39ff9a;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.img-label-gray {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.7rem;
    color: #6a8a9a;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
[data-testid="stNumberInput"] input {
    background-color: #0d1a12 !important;
    border: 1px solid #1a3a26 !important;
    color: #c8d6e5 !important;
    font-family: 'Share Tech Mono', monospace !important;
    border-radius: 2px !important;
}
label, .stCheckbox label {
    color: #8aaa9a !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.08em !important;
}
.stButton > button {
    background: #0a2e1a !important;
    color: #39ff9a !important;
    border: 1px solid #39ff9a !important;
    border-radius: 2px !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    padding: 0.6rem 1.5rem !important;
    transition: background 0.2s, color 0.2s;
}
.stButton > button:hover {
    background: #39ff9a !important;
    color: #0b0e14 !important;
}
.stDownloadButton > button {
    background: #0d1a12 !important;
    color: #6a9a7a !important;
    border: 1px solid #1a3a26 !important;
    border-radius: 2px !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.1em !important;
    width: 100% !important;
}
.stDownloadButton > button:hover {
    border-color: #39ff9a !important;
    color: #39ff9a !important;
}
[data-testid="stSidebar"] h2 {
    font-family: 'Share Tech Mono', monospace !important;
    color: #39ff9a !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
}
[data-testid="stAlert"] {
    background-color: #0d1a12 !important;
    border: 1px solid #1a3a26 !important;
    border-radius: 2px !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.78rem !important;
}
hr { border-color: #1f2d3d !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown('<div class="sar-header">🛰 SAR Image Processor</div>', unsafe_allow_html=True)
st.markdown('<div class="sar-sub">// SYNTHETIC APERTURE RADAR · DESPECKLE & COLORIZE PIPELINE</div>', unsafe_allow_html=True)

badge_cls = "badge-gpu" if DEVICE == "cuda" else "badge-cpu"
st.markdown(f'<span class="badge {badge_cls}">◉ COMPUTE: {DEVICE.upper()}</span>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Model definitions
# ─────────────────────────────────────────────
class Inception(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        bc1 = out_channels // 3
        bc2 = out_channels // 3
        bc3 = out_channels - (bc1 + bc2)
        self.kernel_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, bc1, 1), nn.BatchNorm2d(bc1), nn.ReLU())
        self.kernel_3x3 = nn.Sequential(
            nn.Conv2d(in_channels, bc2, 1), nn.BatchNorm2d(bc2), nn.ReLU(),
            nn.Conv2d(bc2, bc2, 3, padding=1), nn.BatchNorm2d(bc2), nn.ReLU())
        self.pooling = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, bc3, 1), nn.BatchNorm2d(bc3), nn.ReLU())

    def forward(self, x):
        return torch.cat([self.kernel_1x1(x), self.kernel_3x3(x), self.pooling(x)], dim=1)


class ID_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.convL1     = nn.Conv2d(1,  64, 3, padding=1)
        self.inception1 = Inception(64, 64)
        self.inception2 = Inception(64, 64)
        self.inception3 = Inception(64, 64)
        self.convL2     = nn.Conv2d(64, 64, 3, padding=1)
        self.convL3     = nn.Conv2d(64, 64, 3, padding=1)
        self.convL4     = nn.Conv2d(64, 64, 3, padding=1)
        self.convL5     = nn.Conv2d(64, 64, 3, padding=1)
        self.convL6     = nn.Conv2d(64, 64, 3, padding=1)
        self.convL7     = nn.Conv2d(64, 64, 3, padding=1)
        self.convL8     = nn.Conv2d(64,  1, 3, padding=1)
        self.BatchNorm2 = nn.BatchNorm2d(64)
        self.BatchNorm3 = nn.BatchNorm2d(64)
        self.BatchNorm4 = nn.BatchNorm2d(64)
        self.BatchNorm5 = nn.BatchNorm2d(64)
        self.BatchNorm6 = nn.BatchNorm2d(64)
        self.BatchNorm7 = nn.BatchNorm2d(64)
        self.relu       = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.convL1(x))
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.inception3(x)
        x = self.relu(self.BatchNorm2(self.convL2(x)))
        x = self.relu(self.BatchNorm3(self.convL3(x)))
        x = self.relu(self.BatchNorm4(self.convL4(x)))
        x = self.relu(self.BatchNorm5(self.convL5(x)))
        x = self.relu(self.BatchNorm6(self.convL6(x)))
        x = self.relu(self.BatchNorm7(self.convL7(x)))
        return torch.clamp(self.relu(self.convL8(x)), min=0.01)


@st.cache_resource
def load_despeckler(weights_path, device):
    model = ID_CNN().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model


def despeckle(img_pil, model, resize, device):
    original_size = img_pil.size
    tfm_list = []
    if resize:
        tfm_list.append(transforms.Resize(resize))
    tfm_list += [transforms.Grayscale(1), transforms.ToTensor()]
    tfm = transforms.Compose(tfm_list)
    x = tfm(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        out = torch.clamp(x / (model(x) + 1e-8), 0, 1)
    denoised = (out.squeeze().cpu().numpy() * 255).astype(np.uint8)
    result = Image.fromarray(denoised, mode='L')
    if resize:
        result = result.resize(original_size, Image.BILINEAR)
    return result


def build_generator(size, device):
    body  = create_body(resnet18(), pretrained=False, n_in=1, cut=-2)
    net_G = DynamicUnet(body, n_out=2, img_size=(size, size),
                        self_attention=True, act_cls=nn.ReLU)
    return net_G.to(device)


@st.cache_resource
def load_colorizer(weights_path, size, device):
    net_G = build_generator(size, device)
    net_G.load_state_dict(torch.load(weights_path, map_location=device))
    net_G.eval()
    return net_G


def lab_to_rgb(L, ab, ab_saturation=1.0):
    """L (1,H,W) [-1,1], ab (2,H,W) [-1,1] → RGB uint8 (H,W,3).
    ab_saturation: scale factor for ab channels (1.0 = full, 0.0 = grayscale).
    Reducing this below 1.0 dampens the red/magenta tinge from model bias.
    """
    L_lab = (L.numpy()[0] + 1.0) * 127.5 / 255.0 * 100.0
    ab_np = ab.numpy() * 110.0 * ab_saturation
    Lab   = np.stack([L_lab, ab_np[0], ab_np[1]], axis=-1)
    return (np.clip(lab2rgb(Lab), 0, 1) * 255).astype("uint8")


def colorize(denoised_pil, net_G, size, device, ab_saturation=1.0):
    original_size = denoised_pil.size
    sar_resized   = transforms.functional.resize(denoised_pil, (size, size))
    L   = (np.array(sar_resized, dtype="float32")[:, :, np.newaxis] / 127.5) - 1.0
    L_t = torch.from_numpy(L).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        ab_t = net_G(L_t).squeeze(0).cpu()
    rgb = lab_to_rgb(L_t.squeeze(0).cpu(), ab_t, ab_saturation)
    return Image.fromarray(rgb).resize(original_size, Image.BILINEAR)


# ─────────────────────────────────────────────
# Sidebar — image upload + settings
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙ PIPELINE CONFIG")
    st.markdown('<div class="section-label">// INPUT IMAGE</div>', unsafe_allow_html=True)
    uploaded_image = st.file_uploader(
        "Upload SAR image", type=["tiff", "tif", "png", "jpg", "jpeg"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown('<div class="section-label">// DESPECKLE</div>', unsafe_allow_html=True)
    despeckle_resize_w = st.number_input("Resize width",  min_value=64, max_value=2048, value=512, step=32)
    despeckle_resize_h = st.number_input("Resize height", min_value=64, max_value=2048, value=512, step=32)

    st.markdown("---")
    st.markdown('<div class="section-label">// COLORIZER</div>', unsafe_allow_html=True)
    colorizer_size = st.number_input("Input size (square)", min_value=64, max_value=1024, value=256, step=32)
    ab_saturation  = st.slider(
        "Color saturation",
        min_value=0.0, max_value=1.0, value=0.5, step=0.05,
        help="Scales the ab (color) channels. Reduce to correct red/magenta model bias.",
    )

    st.markdown("---")
    st.markdown('<div class="section-label">// STAGES</div>', unsafe_allow_html=True)
    run_despeckle = st.checkbox("Despeckle", value=True)
    run_colorize  = st.checkbox("Colorize",  value=True)

    st.markdown("---")
    st.markdown('<div class="section-label">// MODEL PATHS</div>', unsafe_allow_html=True)
    st.code(
        f"DSPK: {Path(DESPECKLE_WEIGHTS).name}\n"
        f"CLRZ: {Path(COLORIZER_WEIGHTS).name}",
        language=None,
    )

    run_button = st.button("▶  RUN PIPELINE", use_container_width=True)

# ─────────────────────────────────────────────
# Idle state
# ─────────────────────────────────────────────
if not run_button:
    if not uploaded_image:
        st.markdown("""
        <div style="margin-top:5rem; text-align:center; opacity:0.3;">
            <div style="font-family:'Share Tech Mono',monospace; font-size:5rem; line-height:1; color:#39ff9a;">◎</div>
            <div style="font-family:'Share Tech Mono',monospace; font-size:0.75rem; letter-spacing:0.3em; margin-top:1rem;">
                AWAITING INPUT
            </div>
            <div style="font-family:'Share Tech Mono',monospace; font-size:0.65rem; letter-spacing:0.15em; color:#4a6a5a; margin-top:0.5rem;">
                Upload a SAR image in the sidebar → press RUN
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="margin-top:5rem; text-align:center; opacity:0.3;">
            <div style="font-family:'Share Tech Mono',monospace; font-size:5rem; line-height:1; color:#39ff9a;">◉</div>
            <div style="font-family:'Share Tech Mono',monospace; font-size:0.75rem; letter-spacing:0.3em; margin-top:1rem;">
                IMAGE LOADED — READY
            </div>
            <div style="font-family:'Share Tech Mono',monospace; font-size:0.65rem; letter-spacing:0.15em; color:#4a6a5a; margin-top:0.5rem;">
                Press ▶ RUN PIPELINE to begin
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.stop()

if not uploaded_image:
    st.error("Please upload a SAR image before running.")
    st.stop()

# ─────────────────────────────────────────────
# Pipeline execution
# ─────────────────────────────────────────────
raw_pil = Image.open(uploaded_image).convert('L')
raw_arr = np.array(raw_pil, dtype="float32")
raw_display = ((raw_arr - raw_arr.min()) / (raw_arr.max() - raw_arr.min() + 1e-8) * 255).astype("uint8")

denoised_pil  = None
colorized_pil = None

progress = st.progress(0, text="Initialising...")

if run_despeckle:
    progress.progress(10, text="Loading despeckler weights...")
    despeckle_model = load_despeckler(DESPECKLE_WEIGHTS, DEVICE)
    progress.progress(30, text="Despeckling image...")
    denoised_pil = despeckle(raw_pil, despeckle_model, (despeckle_resize_h, despeckle_resize_w), DEVICE)
    progress.progress(55, text="Despeckling complete.")
else:
    denoised_pil = raw_pil

if run_colorize:
    progress.progress(60, text="Loading colorizer weights...")
    colorizer_model = load_colorizer(COLORIZER_WEIGHTS, int(colorizer_size), DEVICE)
    progress.progress(75, text="Colorizing image...")
    colorized_pil = colorize(denoised_pil, colorizer_model, int(colorizer_size), DEVICE, ab_saturation)
    progress.progress(100, text="Pipeline complete.")
else:
    progress.progress(100, text="Pipeline complete.")

# ─────────────────────────────────────────────
# Results display
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-label">// OUTPUT FRAMES</div>', unsafe_allow_html=True)

images = [("RAW SAR", Image.fromarray(raw_display), "gray")]
if run_despeckle:
    images.append(("DESPECKLED", denoised_pil, "gray"))
if run_colorize:
    images.append(("COLORIZED", colorized_pil, "rgb"))

cols = st.columns(len(images), gap="medium")
for col, (title, img, mode) in zip(cols, images):
    label_cls = "img-label" if mode == "rgb" else "img-label-gray"
    col.markdown(f'<div class="{label_cls}">{title}</div>', unsafe_allow_html=True)
    col.image(img, use_container_width=True, clamp=True)

# ─────────────────────────────────────────────
# Downloads
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-label">// EXPORT</div>', unsafe_allow_html=True)

stem = Path(uploaded_image.name).stem
dl_cols = st.columns(len(images), gap="medium")
for col, (title, img, _) in zip(dl_cols, images):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    col.download_button(
        label=f"⬇ {title}.png",
        data=buf,
        file_name=f"{stem}_{title.lower()}.png",
        mime="image/png",
        use_container_width=True,
    )