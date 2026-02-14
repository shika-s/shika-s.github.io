import io
import json
from pathlib import Path
from typing import List, Tuple
import numpy as np
import streamlit as st
import librosa
import soundfile as sf
from PIL import Image
import re
from typing import List, Tuple, Optional


# Optional / lazy imports for frameworks
_torch = None
_tf = None

st.set_page_config(page_title="Species Classifier (Keras/PyTorch)", layout="wide")

# -----------------------------
# CONFIG
# -----------------------------
DEFAULT_SR = 22050
WINDOW_SECONDS = 5.0
N_MELS = 128
FMIN = 20
FMAX = None
IMAGE_SIZE = 224
HOP_LENGTH = 512
N_FFT = 2048

# ---- paths ----
from pathlib import Path
BASE_DIR = Path(__file__).parent
MODEL_FILE = BASE_DIR / "model.keras"
CLASS_MAP_FILE = BASE_DIR / "class_labels.json"

# ---- globals (initialize BEFORE usage) ----
model = None
class_names = None
backend = None
device = None


# -----------------------------
# CLASS MAP
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_class_map(path: str = CLASS_MAP_FILE) -> List[str]:
    try:
        with open(path, "r") as f:
            data = json.load(f)
        # Accept list or {index: name} dict
        if isinstance(data, dict):
            # Map by sorted numeric keys
            keys = sorted(map(int, data.keys()))
            return [data[str(k)] if str(k) in data else data[k] for k in keys]
        elif isinstance(data, list):
            return data
    except Exception as e:
        st.warning(f"Could not read class map: {e}. Falling back to defaults.")
    return ["Amphibia", "Aves", "Insecta", "Mammalia"]

def _build_image_array(img: Image.Image, image_size: int = IMAGE_SIZE) -> np.ndarray:
    img = img.resize((image_size, image_size))
    arr = np.asarray(img).astype("float32") / 255.0  # [0,1]
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    return arr

# -----------------------------
# AUDIO ⇨ MEL IMAGE
# -----------------------------
def load_audio(file_bytes: bytes, sr: int = DEFAULT_SR) -> Tuple[np.ndarray, int]:
    y, orig_sr = sf.read(io.BytesIO(file_bytes), dtype='float32')
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if orig_sr != sr:
        try:
            y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)
        except TypeError:
            # For librosa >= 0.10
            y = librosa.resample(y, orig_sr=orig_sr, sr=sr)
    return y, sr

def frame_audio(y: np.ndarray, sr: int, seconds: float):
    hop = int(seconds * sr)
    frames = []
    for start in range(0, len(y), hop):
        end = start + hop
        chunk = y[start:end]
        if len(chunk) < hop:
            chunk = np.pad(chunk, (0, hop - len(chunk)), mode='constant')
        frames.append(chunk)
    return frames

def audio_to_mel_image(y: np.ndarray, sr: int) -> Image.Image:
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    S_db_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)
    img = (S_db_norm * 255.0).astype(np.uint8)
    img3 = np.stack([img, img, img], axis=-1)
    return Image.fromarray(img3)

@st.cache_resource(show_spinner=True)
def load_model_auto(model_path: str):
    """Load either a Keras (.keras/.h5) or PyTorch (.pt/.pth) model based on file extension."""
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    suffix = p.suffix.lower()
    class_names = load_class_map()

    # -----------------
    # KERAS MODELS
    # -----------------
    if suffix in {".keras", ".h5"}:
        import tensorflow as tf

        # Try to cover common preprocess functions
        custom_objects = {}
        try:
            from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
            custom_objects["resnet_preprocess"] = resnet_preprocess
            custom_objects["preprocess_input"] = resnet_preprocess
        except ImportError:
            pass
        try:
            from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess
            custom_objects["efficientnet_preprocess"] = effnet_preprocess
        except ImportError:
            pass
        try:
            from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
            custom_objects["inception_preprocess"] = inception_preprocess
        except ImportError:
            pass

        try:
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        except Exception as e:
            raise RuntimeError(f"Failed to load Keras model: {e}")

        backend = "keras"
        device = "CPU"
        return model, class_names, backend, device

    # -----------------
    # PYTORCH MODELS
    # -----------------
    import torch
    import torch.nn as nn
    import torchvision.models as tvm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Try TorchScript
    try:
        model = torch.jit.load(model_path, map_location=device)
        model.eval().to(device)
        backend = "torch"
        return model, class_names, backend, str(device)
    except Exception:
        pass

    # Try state_dict
    obj = torch.load(model_path, map_location=device)
    if isinstance(obj, dict) and "state_dict" in obj:
        backbone = tvm.resnet50(weights=None)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, len(class_names))
        backbone.load_state_dict(obj["state_dict"])
        model = backbone
    elif isinstance(obj, dict):  # plain state_dict
        backbone = tvm.resnet50(weights=None)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, len(class_names))
        backbone.load_state_dict(obj)
        model = backbone
    elif isinstance(obj, nn.Module):
        model = obj
    else:
        raise RuntimeError("Unrecognized PyTorch model format.")

    model.eval().to(device)
    backend = "torch"
    return model, class_names, backend, str(device)

# -----------------------------
# INFERENCE
# -----------------------------
def predict_img(model, backend: str, device, img: Image.Image, class_names: List[str], x_meta: Optional[np.ndarray] = None) -> tuple[str, float, np.ndarray]:
    arr = _build_image_array(img, IMAGE_SIZE)

    if backend == "keras":
        x_img = np.expand_dims(arr, axis=0)  # (1, H, W, 3)
        # If model expects 2+ inputs, feed metadata too
        expects_multi = hasattr(model, "inputs") and isinstance(model.inputs, (list, tuple)) and len(model.inputs) >= 2
        if expects_multi:
            # If user didn't provide metadata, try zeros with inferred dim
            if x_meta is None:
                dim = get_meta_dim_from_model(model)
                if dim is None:
                    raise ValueError("Model expects metadata input but no metadata provided, and its length couldn't be inferred.")
                x_meta = np.zeros((1, dim), dtype="float32")
            preds = model.predict([x_img, x_meta], verbose=0)
        else:
            preds = model.predict(x_img, verbose=0)
        preds = np.squeeze(preds).astype("float32")

    else:
        torch, _, _ = _torch
        import torchvision.transforms as T
        transform = T.Compose([T.Resize((IMAGE_SIZE, IMAGE_SIZE)), T.ToTensor()])
        x = transform(img).unsqueeze(0).to(device)
        with torch.inference_mode():
            logits = model(x)
            preds = torch.softmax(logits, dim=1).cpu().numpy().squeeze()

    top = int(np.argmax(preds))
    return class_names[top], float(preds[top]), preds


def summarize_votes(votes):
    if not votes:
        return "N/A", 0.0
    from collections import Counter
    c = Counter(votes)
    label, count = c.most_common(1)[0]
    return label, count / len(votes)

def get_meta_dim_from_model(model) -> Optional[int]:
    """Try to infer the metadata vector length from the 2nd Keras input."""
    try:
        # KerasTensor shape like (None, D)
        shp = model.inputs[1].shape
        dim = shp[-1]
        return int(dim) if dim is not None else None
    except Exception:
        return None

def parse_meta_vector(text: str, meta_dim: int, default: float = 0.0) -> np.ndarray:
    """Parse comma/space-separated floats into shape (1, meta_dim), pad/truncate as needed."""
    if not text or not text.strip():
        return np.full((1, meta_dim), default, dtype="float32")
    tokens = [t for t in re.split(r"[,\s]+", text.strip()) if t]
    vals = []
    for t in tokens:
        try:
            vals.append(float(t))
        except Exception:
            vals.append(default)
    if len(vals) < meta_dim:
        vals += [default] * (meta_dim - len(vals))
    elif len(vals) > meta_dim:
        vals = vals[:meta_dim]
    return np.asarray(vals, dtype="float32").reshape(1, meta_dim)

@st.cache_data(show_spinner=True)
def fetch_model(url: str, dst: str = "model.keras") -> str:
    import os, re, urllib.request

    def _extract_drive_id(u: str) -> str | None:
        # Handles /file/d/<id>/..., ?id=<id>, uc?id=<id>
        patterns = [
            r"drive\.google\.com/file/d/([^/]+)/",
            r"drive\.google\.com/.*?[?&]id=([^&]+)",
            r"docs\.google\.com/uc\?export=download&id=([^&]+)",
        ]
        for p in patterns:
            m = re.search(p, u)
            if m:
                return m.group(1)
        return None

    def _to_drive_direct(u: str) -> str:
        file_id = _extract_drive_id(u)
        return f"https://drive.google.com/uc?export=download&id={file_id}" if file_id else u

    if os.path.exists(dst):
        return dst

    # If it's a Drive link, prefer gdown (handles confirm tokens for big files)
    if "drive.google.com" in url or "docs.google.com" in url:
        try:
            import gdown  # add `gdown==5.2.0` to requirements.txt
            file_id = _extract_drive_id(url)
            if file_id:
                gdown.download(id=file_id, output=dst, quiet=False)
                if os.path.exists(dst):
                    return dst
        except Exception as _:
            pass  # fall back to direct

        # Fallback to direct uc link
        url = _to_drive_direct(url)

    # Generic download
    urllib.request.urlretrieve(url, dst)
    return dst

# In Settings panel (after MODEL_FILE = st.text_input(...)):
if "MODEL_URL" in st.secrets and not Path(MODEL_FILE).exists():
    with st.spinner("Downloading model..."):
        MODEL_FILE = fetch_model(st.secrets["MODEL_URL"])

# -----------------------------
# UI
# -----------------------------
st.title("🎶🐦 Species Classifier from Audio")
st.write("Upload an audio file. We'll chop it into 5-second segments, convert to mel-spectrograms, and run a CNN model.")

with st.expander("⚙️ Settings", expanded=False):
    DEFAULT_SR = st.number_input("Resample rate (Hz)", value=DEFAULT_SR, step=1000)
    WINDOW_SECONDS = st.number_input("Window length (seconds)", value=float(WINDOW_SECONDS), step=1.0, min_value=1.0, max_value=30.0)
    IMAGE_SIZE = st.number_input("Image size (pixels)", value=int(IMAGE_SIZE), step=16, min_value=64, max_value=512)
    N_MELS = st.number_input("Number of Mel bands", value=int(N_MELS), step=8, min_value=32, max_value=256)
    scan_full = st.checkbox("Scan entire audio in 5s windows (majority vote)", value=True, key="scan_full")
    show_spectrograms = st.checkbox("Show spectrogram thumbnails", value=True)
    MODEL_FILE = st.text_input("Model file path (.keras/.h5 for Keras, .pt/.pth for PyTorch)", value=MODEL_FILE)

if Path(MODEL_FILE).exists():
    try:
        model, class_names, backend, device = load_model_auto(MODEL_FILE)
        st.success(f"Loaded {backend.upper()} model")
    except Exception as e:
        st.error(f"Error loading model: {e}")

# =========================
# METADATA UI (2nd input)
# =========================
x_meta = None
if model is not None and backend == "keras":
    multi_input = hasattr(model, "inputs") and isinstance(model.inputs, (list, tuple)) and len(model.inputs) >= 2
    if multi_input:
        expected_dim = get_meta_dim_from_model(model)  # from earlier helper; returns int or None

        with st.expander("🧮 Metadata (2nd input)", expanded=True):
            st.write("Set metadata for this audio segment:")

            # Raw inputs for all 5 fields
            call_toggle = st.toggle("Call", value=False, help="True if this is a 'call' vocalization")
            song_toggle = st.toggle("Song / Canto", value=False, help="True if this is song/canto")
            category_val = st.number_input("Category (int)", value=1, step=1, min_value=0, help="Integer class/category code")
            lat_val = st.number_input("Latitude", value=6.2551, step=0.0001, format="%.6f", min_value=-90.0, max_value=90.0)
            lon_val = st.number_input("Longitude", value=-75.5153, step=0.0001, format="%.6f", min_value=-180.0, max_value=180.0)

            # Build dictionary for convenience
            metadata_values = {
                "call": float(call_toggle),
                "category": float(category_val),
                "song/canto": float(song_toggle),
                "latitude": float(lat_val),
                "longitude": float(lon_val),
            }

            fields_all = ["call", "category", "song/canto", "latitude", "longitude"]

            # Determine active fields to match model's expected dim
            if expected_dim is None:
                # Fall back to using all 5 if unknown
                active_fields = fields_all
                st.info("Could not infer metadata length from the model; passing all 5 fields.")
            else:
                if expected_dim == 5:
                    active_fields = fields_all
                elif expected_dim < 5:
                    # Let the user choose which field to drop (default: drop 'category')
                    drop_default = "category" if "category" in fields_all else fields_all[0]
                    drop_field = st.selectbox(
                        f"Model expects {expected_dim} metadata features. Select ONE field to drop:",
                        options=fields_all,
                        index=fields_all.index(drop_default)
                    )
                    remaining = [f for f in fields_all if f != drop_field]
                    # If still more than expected_dim (e.g., expected_dim==3), drop from the middle (or let user reorder in future)
                    active_fields = remaining[:expected_dim]
                    st.caption(f"Using fields: {active_fields}")
                else:
                    # expected_dim > 5 — pad with zeros for extra dims (rare)
                    active_fields = fields_all
                    st.warning(
                        f"Model expects {expected_dim} features but only 5 are provided. "
                        "We'll pad the remainder with zeros."
                    )

            # Assemble the vector in the chosen order
            x_list = [metadata_values[f] for f in active_fields]
            # Pad if model wants more than 5
            if expected_dim is not None and expected_dim > len(x_list):
                x_list += [0.0] * (expected_dim - len(x_list))

            x_meta = np.array([x_list], dtype="float32")

            st.caption(
                f"Metadata vector -> shape {x_meta.shape}: {x_list}"
            )

uploaded = st.file_uploader("Audio file (.wav, .mp3, .flac, .ogg)", type=["wav", "mp3", "flac", "ogg"])

if uploaded is not None and model is not None:
    st.audio(uploaded, format=uploaded.type)
    try:
        with st.spinner("🔄 Running model inference... Please wait."):
            y, sr = load_audio(uploaded.read(), sr=DEFAULT_SR)
            chunks = frame_audio(y, sr, WINDOW_SECONDS) if scan_full else [y[:int(WINDOW_SECONDS*sr)]]

            votes = []
            all_probs = []

            st.write(f"🔪 Chunks: {len(chunks)} × {WINDOW_SECONDS:.0f}s")

            for i, chunk in enumerate(chunks):
                img = audio_to_mel_image(chunk, sr)
                label, conf, probs = predict_img(model, backend, device, img, class_names, x_meta=x_meta)
                votes.append(label)
                all_probs.append(probs)

                if show_spectrograms:
                    with st.expander(f"Segment {i+1}: {label} ({conf*100:.1f}%)", expanded=False):
                        st.image(img, caption=f"Mel-Spectrogram (Segment {i+1})", use_container_width=True)

            final_label, agreement = summarize_votes(votes)
            st.subheader("Final Prediction")
            st.write(f"**{final_label}**  (agreement: {agreement*100:.1f}% across segments)")

            if all_probs:
                mean_probs = np.mean(np.stack(all_probs, axis=0), axis=0)
                order = np.argsort(mean_probs)[::-1]
                st.write("Top predictions (averaged):")
                for idx in order[:min(5, len(class_names))]:
                    st.write(f"- {class_names[idx]}: {mean_probs[idx]*100:.1f}%")

    except Exception as e:
        st.error(f"Error processing audio: {e}")
        st.exception(e)
