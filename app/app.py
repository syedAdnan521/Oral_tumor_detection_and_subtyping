# app/streamlit_oralpatho_full.py
import streamlit as st
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import io, json, zipfile, time
import matplotlib.pyplot as plt
import base64
from typing import Tuple, List

st.set_page_config(layout="wide", page_title="AI Tumour Detection — OralPatho Demo")

# -------------------------- CONFIG (edit paths if needed) --------------------------
PROJECT_ROOT = Path("/Users/syedadnanahmad/Downloads/AI_Tumour_detection")
EMB_ROOT     = PROJECT_ROOT / "embeddings"
MODEL_DIR    = PROJECT_ROOT / "models"
PATCH_IMG_ROOT = PROJECT_ROOT / "data"    # optional: where patches might be stored (searchable)
RESULTS_DIR  = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# default model filenames (must match your saved checkpoints)
STAGE1_MODEL = MODEL_DIR / "stage1_balanced_best.pth"
STAGE2_MODEL = MODEL_DIR / "oscc_topkK20_best.pth"

# constants — update if different
K_DEFAULT = 20
OSCC_CLASSES = ["wdoscc","mdoscc","pdoscc"]
FIVE_CLASSES = ["normal","osmf","wdoscc","mdoscc","pdoscc"]
# -----------------------------------------------------------------------------------

# -------------------------- DEVICE --------------------------
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
# -----------------------------------------------------------------------------------

# -------------------------- MODEL DEFINITION (must match training) --------------------------
class TopK_AttentionMIL(nn.Module):
    def __init__(self, emb_dim=512, hidden_dim=512, num_classes=3, k=20, dropout_p=0.5):
        super().__init__()
        self.k = k
        self.V = nn.Linear(emb_dim, hidden_dim)
        self.U = nn.Linear(emb_dim, hidden_dim)
        self.w = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, emb_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(emb_dim//2, num_classes)
        )

    def forward(self, H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # H: (N, D)
        Vh = torch.tanh(self.V(H))
        Uh = torch.sigmoid(self.U(H))
        A = self.w(Vh * Uh).squeeze(1)   # (N,)
        Ksel = min(self.k, H.shape[0])
        top_vals, top_idx = torch.topk(A, Ksel)
        H_top = H[top_idx]               # (Ksel, D)
        A_top = torch.softmax(top_vals, dim=0)  # (Ksel,)
        agg = torch.sum(A_top.unsqueeze(1) * H_top, dim=0)  # (D,)
        logits = self.classifier(agg).unsqueeze(0)  # (1, C)
        return logits, A_top.detach().cpu(), top_idx.detach().cpu()
# -----------------------------------------------------------------------------------

# -------------------------- HELPERS --------------------------
@st.cache_resource
def load_model(path: Path, emb_dim=512, hidden_dim=512, num_classes=3, k=K_DEFAULT, dropout_p=0.5):
    if not path.exists():
        return None
    m = TopK_AttentionMIL(emb_dim=emb_dim, hidden_dim=hidden_dim, num_classes=num_classes, k=k, dropout_p=dropout_p)
    ck = torch.load(path, map_location=DEVICE)
    # handle either dict or raw state_dict
    if isinstance(ck, dict) and "model_state" in ck:
        m.load_state_dict(ck["model_state"])
    else:
        m.load_state_dict(ck)
    m.eval()
    m.to(DEVICE)
    return m

def run_inference_single(model: TopK_AttentionMIL, emb_np: np.ndarray, k: int):
    """Runs one deterministic forward pass. Returns probs (1D), attn (1D topK), top_idx (1D)."""
    H = torch.from_numpy(emb_np.astype(np.float32)).to(DEVICE)
    # if model.k differs from k, replace temporarily
    orig_k = model.k
    model.k = k
    with torch.no_grad():
        logits, attn, top_idx = model(H)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    model.k = orig_k
    return probs, attn.numpy(), top_idx.numpy()

def mc_dropout_inference(model: TopK_AttentionMIL, emb_np: np.ndarray, k: int, T=20):
    """Run T stochastic forward passes (MC Dropout). model should be in train() mode to enable dropout."""
    H = torch.from_numpy(emb_np.astype(np.float32)).to(DEVICE)
    orig_k = model.k
    model.k = k
    model.train()   # enable dropout
    preds = []
    attn_list = []
    topidx_list = []
    for _ in range(T):
        logits, attn, top_idx = model(H)
        p = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
        preds.append(p)
        attn_list.append(attn.numpy())
        topidx_list.append(top_idx.numpy())
    model.eval()
    model.k = orig_k
    preds = np.vstack(preds)   # (T, C)
    mean_p = preds.mean(axis=0)
    var_p = preds.var(axis=0)
    entropy = -np.sum(mean_p * np.log(mean_p + 1e-12))
    # majority / consensus top idx across runs (optional)
    return mean_p, var_p, entropy, attn_list, topidx_list

def find_patch_images_for_slide(slide_id: str, top_indices: List[int], max_show:int=9) -> List[Path]:
    """
    Naively search PATCH_IMG_ROOT for images containing slide_id substring.
    Then try to map top_indices to filenames by searching for numbers or patch idx in name.
    If not found, return the first matching images up to max_show.
    """
    results = []
    if not PATCH_IMG_ROOT.exists():
        return results
    # collect candidates that include slide_id in filename
    imgs = list(PATCH_IMG_ROOT.rglob(f"*{slide_id}*.png")) + list(PATCH_IMG_ROOT.rglob(f"*{slide_id}*.jpg"))
    if not imgs:
        # fallback: return any images in folder (none likely)
        return []
    # try to pick images that contain the top index as substring (e.g., slide-12_patch_5.png)
    for idx in top_indices:
        found = None
        for p in imgs:
            name = p.name.lower()
            if f"{idx}" in name:
                found = p; break
        if found:
            results.append(found)
        if len(results) >= max_show:
            break
    # if no mapping, return first few images
    if not results:
        results = imgs[:max_show]
    return results

def render_prob_bars(probs: np.ndarray, class_names: List[str], width=0.4):
    fig, ax = plt.subplots(figsize=(5,2.2))
    y = range(len(class_names))
    ax.barh(y, probs, height=0.6)
    ax.set_yticks(y); ax.set_yticklabels(class_names)
    ax.set_xlim(0,1)
    for i, v in enumerate(probs):
        ax.text(v + 0.01, i, f"{v:.2f}", va='center')
    ax.invert_yaxis()
    ax.set_xlabel("Probability")
    plt.tight_layout()
    return fig

def make_downloadable_zip(slide_id: str, emb_path: Path, probs, attn, top_idx, thumbnail_paths: List[Path], meta: dict):
    """Create an in-memory zip containing JSON + thumbnails + small report image and return bytes."""
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, mode="w") as z:
        # JSON
        j = {"slide_id": slide_id, "probs": probs.tolist(), "attn": attn.tolist() if hasattr(attn, "tolist") else list(attn),
             "top_idx": top_idx.tolist() if hasattr(top_idx, "tolist") else list(top_idx), "meta": meta}
        z.writestr(f"{slide_id}_result.json", json.dumps(j, indent=2))
        # add thumbnails
        for i, p in enumerate(thumbnail_paths):
            try:
                z.write(p, arcname=f"thumb_{i}_{p.name}")
            except Exception:
                pass
    bio.seek(0)
    return bio.getvalue()

def st_download_button_bytes(label, bts, filename):
    b64 = base64.b64encode(bts).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">{label}</a>'
    st.markdown(href, unsafe_allow_html=True)
# -----------------------------------------------------------------------------------

# -------------------------- UI Layout --------------------------
st.title("AI Based Tumour Detection and Subtyping")
st.sidebar.header("Settings")

# sidebar model toggles & K, MC options
use_stage1 = st.sidebar.checkbox("Use Stage-1 (Normal vs OSCC)", value=True)
use_stage2 = st.sidebar.checkbox("Use Stage-2 (OSCC subtyping)", value=True)
k_val = st.sidebar.number_input("Top-K / bag size (K)", min_value=1, max_value=200, value=K_DEFAULT, step=1)
mc_enabled = st.sidebar.checkbox("Enable MC-Dropout uncertainty (T runs)", value=False)
mc_T = st.sidebar.slider("MC T (runs)", 5, 50, 20, step=5) if mc_enabled else 1
uncertainty_threshold = st.sidebar.slider("Entropy threshold (flag uncertain if > )", 0.0, 2.0, 1.0, step=0.05)

st.sidebar.markdown("---")
st.sidebar.write("Model paths (edit if needed)")
s1_path = st.sidebar.text_input("Stage1 model path", str(STAGE1_MODEL))
s2_path = st.sidebar.text_input("Stage2 model path", str(STAGE2_MODEL))
st.sidebar.write("Patch image root (for thumbnails):")
st.sidebar.text_input("PATCH_IMG_ROOT", str(PATCH_IMG_ROOT))

# load models (cached)
model_stage1 = load_model(Path(s1_path), emb_dim=512, hidden_dim=256, num_classes=2, k=k_val, dropout_p=0.5) if use_stage1 else None
model_stage2 = load_model(Path(s2_path), emb_dim=512, hidden_dim=512, num_classes=3, k=k_val, dropout_p=0.5) if use_stage2 else None

col1, col2 = st.columns([1,2])

with col1:
    # st.subheader("Input embedding")
    # mode = st.radio("Input selector", ["Choose existing embedding", "Upload .npy embedding"], index=0)
    selected_emb_path = None
    # if mode == "Choose existing embedding":
    #     emb_files = list(EMB_ROOT.rglob("*.npy"))
    #     emb_map = {str(p.relative_to(EMB_ROOT)): p for p in emb_files}
    #     selection = st.selectbox("Choose embedding (relative to embeddings/)", ["-- none --"] + sorted(list(emb_map.keys())))
    #     if selection != "-- none --":
    #         selected_emb_path = emb_map[selection]
    
    up = st.file_uploader("Upload a .npy embedding file (patch x dim)", type=["npy"])
    if up is not None:
            t = up.read()
            arr = np.load(io.BytesIO(t))
            tmp = RESULTS_DIR / f"uploaded_{int(time.time())}.npy"
            np.save(tmp, arr)
            selected_emb_path = tmp

    if selected_emb_path is None:
        st.info("Select or upload an embedding to run inference.")
        st.stop()

    st.write("Embedding:", selected_emb_path.name)
    emb_arr = np.load(selected_emb_path)
    st.write("Shape:", emb_arr.shape)
    # attempt to get slide id from filename
    slide_id_guess = selected_emb_path.stem

    st.subheader("Run controls")
    run_btn = st.button("Run pipeline")
    st.markdown("**Note:** If MC-Dropout enabled, the app will run multiple stochastic forward passes (slower).")

with col2:
    st.subheader("Results")
    placeholder = st.empty()

# -------------------------- Inference & Display --------------------------
if run_btn:
    meta = {"k": int(k_val), "mc_enabled": bool(mc_enabled), "mc_T": int(mc_T), "timestamp": time.time()}
    # Stage-1
    if model_stage1 is not None:
        # deterministic or MC
        if mc_enabled:
            mean_p1, var1, ent1, a_list1, idx_list1 = mc_dropout_inference(model_stage1, emb_arr, k=int(k_val), T=int(mc_T))
            probs1 = mean_p1
            entropy1 = ent1
        else:
            probs1, att1, idx1 = run_inference_single(model_stage1, emb_arr, k=int(k_val))
            entropy1 = -np.sum(probs1 * np.log(probs1 + 1e-12))
            var1 = np.zeros_like(probs1)
            a_list1 = [att1]; idx_list1 = [idx1]
        pred1 = int(np.argmax(probs1))
        is_oscc = (pred1 == 1)
    else:
        probs1 = None; entropy1 = None; is_oscc = True  # if no stage1, continue to stage2

    # Stage-2 if OSCC or no stage1
    if is_oscc and model_stage2 is not None:
        if mc_enabled:
            mean_p2, var2, ent2, a_list2, idx_list2 = mc_dropout_inference(model_stage2, emb_arr, k=int(k_val), T=int(mc_T))
            probs2 = mean_p2; entropy2 = ent2; var2 = var2
            # pick median attention and top_idx from last run for display
            att_display = a_list2[-1]
            idx_display = idx_list2[-1]
        else:
            probs2, att_display, idx_display = run_inference_single(model_stage2, emb_arr, k=int(k_val))
            entropy2 = -np.sum(probs2 * np.log(probs2 + 1e-12))
            var2 = np.zeros_like(probs2)
        pred2 = int(np.argmax(probs2))
    else:
        probs2 = None; att_display = None; idx_display = None; entropy2 = None; pred2 = None

    # Build result dictionary
    result = {
        "slide_id": slide_id_guess,
        "stage1": {
            "probs": probs1.tolist() if probs1 is not None else None,
            "entropy": float(entropy1) if entropy1 is not None else None,
            "pred": int(pred1) if model_stage1 is not None else None
        },
        "stage2": {
            "probs": probs2.tolist() if probs2 is not None else None,
            "entropy": float(entropy2) if entropy2 is not None else None,
            "pred_idx": int(pred2) if pred2 is not None else None,
            "pred_label": OSCC_CLASSES[pred2] if pred2 is not None else None,
            "attn_topk": att_display.tolist() if att_display is not None else None,
            "top_idx": idx_display.tolist() if idx_display is not None else None
        },
        "meta": meta
    }

    # Display Stage1
    with placeholder.container():
        st.markdown("## Pipeline output")
        colA, colB = st.columns([1,2])
        with colA:
            if model_stage1 is not None:
                st.markdown("### Stage-1 (Normal vs OSCC)")
                st.write("Probs:", np.round(probs1,3))
                st.write("Pred:", "OSCC" if int(pred1)==1 else "NORMAL")
                st.write("Entropy:", float(entropy1))
                fig1 = render_prob_bars(probs1, ["normal","oscc"])
                st.pyplot(fig1)
            else:
                st.info("Stage-1 not loaded; skipping to Stage-2")

            st.markdown("### Selected options")
            st.write(f"K = {k_val}")
            st.write("MC-Dropout:", mc_enabled, "T:", mc_T if mc_enabled else "N/A")
            if model_stage1 is not None:
                if entropy1 is not None and entropy1 > uncertainty_threshold:
                    st.warning("Stage-1: LOW CONFIDENCE (entropy > threshold)")

        with colB:
            if model_stage2 is not None and probs2 is not None:
                st.markdown("### Stage-2 (OSCC subtyping)")
                st.write("Classes:", OSCC_CLASSES)
                st.write("Probs:", np.round(probs2,3))
                st.write("Predicted subtype:", OSCC_CLASSES[pred2])
                st.write("Entropy:", float(entropy2))
                fig2 = render_prob_bars(probs2, OSCC_CLASSES)
                st.pyplot(fig2)
                if entropy2 is not None and entropy2 > uncertainty_threshold:
                    st.warning("Stage-2: LOW CONFIDENCE (entropy > threshold)")

                st.markdown("#### Top-K attention & patch indices")
                st.write("Top-K indices (relative to embedding array):", idx_display.tolist())
                st.write("Attention weights (for top-K):", np.round(att_display,3).tolist())

                # thumbnails
                show_thumbs = st.checkbox("Show top-K patch thumbnails (if available)", value=True)
                thumbs = []
                if show_thumbs:
                    thumbs = find_patch_images_for_slide(slide_id_guess, idx_display.tolist(), max_show=min(9,len(idx_display)))
                    if thumbs:
                        cols = st.columns(min(6,len(thumbs)))
                        for i, p in enumerate(thumbs):
                            try:
                                im = Image.open(p).convert("RGB")
                                cols[i % len(cols)].image(im, caption=p.name, use_column_width=True)
                            except Exception as e:
                                cols[i % len(cols)].write(p.name)
                    else:
                        st.info("No patch thumbnails found automatically. If you have patch images, place them under data/ with slide-id in filename.")

    # Export / Download
    st.markdown("---")
    st.subheader("Export results")
    # Save JSON to results
    save_btn = st.button("Save JSON + thumbnails (zip)")
    if save_btn:
        thumbs_list = thumbs if thumbs else []
        zip_bytes = make_downloadable_zip(slide_id_guess, selected_emb_path, np.array(probs2 if probs2 is not None else probs1), att_display if att_display is not None else [], idx_display if idx_display is not None else [], thumbs_list, meta)
        fname = f"{slide_id_guess}_inference_{int(time.time())}.zip"
        st.download_button("Download ZIP", data=zip_bytes, file_name=fname, mime="application/zip")
        # also save JSON single file
        json_path = RESULTS_DIR / f"{slide_id_guess}_inference.json"
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2)
        st.success(f"Saved JSON to {json_path.name} and prepared ZIP.")
