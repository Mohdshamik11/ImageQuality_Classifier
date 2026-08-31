"""
Streamlit UI for the photo-quality classifier + phase-2a enhancement.

Flow:
  1. Upload up to MAX_IMAGES photos.
  2. Each is classified on ingest (tiled -- see src/predict.py) and shown as a
     card. A card's five per-defect scores are hidden until you expand it.
  3. "Enhance flagged photos" runs the classical, flag-driven fixes
     (src/enhance.py) on every photo with at least one flagged defect and shows
     the before/after below.

Run locally:  streamlit run app.py
Deploy:       Streamlit Community Cloud, main file = app.py
"""
import io
import sys
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps

# Reusable logic lives in src/ (same pattern as the notebooks).
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from predict import predict, DEFECT_COLUMNS, THRESHOLD, SHORT_SIDE  # noqa: E402
from enhance import enhance  # noqa: E402

MAX_IMAGES = 15          # RAM cap for the free hosting tier
INGEST_LONG_SIDE = 1400  # downscale bigger uploads so 15 of them fit in memory

BLURB = {
    "blur": "Out of focus or motion-smeared. Scored per region, so a soft background counts.",
    "underexposed": "Too dark overall.",
    "overexposed": "Too bright, highlights blown to white.",
    "noise": "Grainy or speckled, the high-ISO look.",
    "contrast": "Flat and washed out, tones bunched toward mid-grey.",
}

st.set_page_config(
    page_title="Photo Quality Classifier",
    page_icon=":material/photo_camera:",
    layout="wide",
)

# ---- session state ---------------------------------------------------------- #
st.session_state.setdefault("items", {})     # file_id -> {name, image, pred}
st.session_state.setdefault("enhanced", {})  # file_id -> {image, applied}

st.title("Photo quality classifier")
st.caption(
    "Upload photos to flag five kinds of defect (blur, under/over-exposure, noise, low contrast), "
    "then enhance the flagged ones. The model scans the whole frame by tiling a 256-pixel window across it."
)

files = st.file_uploader(
    "Upload photos",
    type=["png", "jpg", "jpeg", "webp"],
    accept_multiple_files=True,
)

if not files:
    st.session_state["items"] = {}
    st.session_state["enhanced"] = {}
    st.info(f"Add up to {MAX_IMAGES} photos to begin.")
    st.stop()

if len(files) > MAX_IMAGES:
    st.warning(f"Showing the first {MAX_IMAGES} of {len(files)} uploads.")
    files = files[:MAX_IMAGES]


def file_key(f) -> str:
    """Stable id for an uploaded file across reruns."""
    return getattr(f, "file_id", None) or f"{f.name}:{f.size}"


# ---- ingest + classify only the NEW files --------------------------------- #
current = {file_key(f) for f in files}
for fid in list(st.session_state["items"]):        # forget files the user removed
    if fid not in current:
        st.session_state["items"].pop(fid, None)
        st.session_state["enhanced"].pop(fid, None)

new = [f for f in files if file_key(f) not in st.session_state["items"]]
if new:
    st.session_state["enhanced"] = {}             # uploads changed -> old enhanced pairs are stale
    bar = st.progress(0.0, text="Scanning new photos...")
    for i, f in enumerate(new, 1):
        try:
            img = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
        except Exception:
            st.warning(f"Could not read {f.name}, skipped.")
            continue
        img.thumbnail((INGEST_LONG_SIDE, INGEST_LONG_SIDE))  # in place; caps memory
        st.session_state["items"][file_key(f)] = {
            "name": f.name, "image": img, "pred": predict(img),
        }
        bar.progress(i / len(new), text=f"Scanning {f.name}  ({i}/{len(new)})")
    bar.empty()

# keep display order == upload order
items = [st.session_state["items"][file_key(f)] for f in files if file_key(f) in st.session_state["items"]]

# ---- card grid ----------------------------------------------------------- #
st.subheader("Uploaded photos")
PER_ROW = 4
for start in range(0, len(items), PER_ROW):
    cols = st.columns(PER_ROW, gap="medium")
    for col, it in zip(cols, items[start:start + PER_ROW]):
        pred = it["pred"]
        flagged = [c for c in DEFECT_COLUMNS if pred["flags"][c]]
        with col.container(border=True):
            st.image(it["image"], width="stretch")
            if flagged:
                st.markdown(f":red[**{len(flagged)} flagged**] &nbsp; " + ", ".join(flagged))
            else:
                st.markdown(":green[**clean**]")
            with st.expander("classifications", type="compact"):
                for c in DEFECT_COLUMNS:
                    p = pred["probs"][c]
                    tag = ":red[flagged]" if pred["flags"][c] else ":gray[clear]"
                    st.markdown(f"**{c}** &nbsp; {p:.0%} &nbsp; {tag}")
                    st.progress(min(max(p, 0.0), 1.0))
                st.caption(f"{pred['n_tiles']} tiles, short side {SHORT_SIDE}px, threshold {THRESHOLD:.2f}")
            st.caption(it["name"])

# ---- enhance ----------------------------------------------------------- #
st.divider()
to_fix = [f for f in files
          if file_key(f) in st.session_state["items"]
          and any(st.session_state["items"][file_key(f)]["pred"]["flags"].values())]

if not to_fix:
    st.info("No defects flagged, so there is nothing to enhance.")
    st.stop()

if st.button(f"Enhance {len(to_fix)} flagged photo(s)", type="primary",
             icon=":material/auto_fix_high:"):
    st.session_state["enhanced"] = {}
    bar = st.progress(0.0, text="Enhancing...")
    for i, f in enumerate(to_fix, 1):
        it = st.session_state["items"][file_key(f)]
        out_img, applied = enhance(it["image"], it["pred"]["flags"], it["pred"]["probs"])
        st.session_state["enhanced"][file_key(f)] = {"image": out_img, "applied": applied}
        bar.progress(i / len(to_fix), text=f"Enhancing {it['name']}  ({i}/{len(to_fix)})")
    bar.empty()

# ---- enhanced results ------------------------------------------------- #
if st.session_state["enhanced"]:
    st.subheader("Enhanced")
    for f in files:
        enh = st.session_state["enhanced"].get(file_key(f))
        if not enh:
            continue
        it = st.session_state["items"][file_key(f)]
        st.markdown(f"**{it['name']}** &nbsp; fixed: {', '.join(enh['applied'])}")
        a, b = st.columns(2, gap="medium")
        a.image(it["image"], caption="original", width="stretch")
        b.image(enh["image"], caption="enhanced", width="stretch")
        buf = io.BytesIO()
        enh["image"].save(buf, format="PNG")
        st.download_button(
            "Download enhanced", buf.getvalue(),
            file_name=f"enhanced_{it['name'].rsplit('.', 1)[0]}.png",
            mime="image/png", key=f"dl_{file_key(f)}", icon=":material/download:",
        )
        st.divider()

st.caption(
    "Model: iteration 4 (`models/traincombo_best.pt`), test macro-F1 0.91 single-defect / 0.88 multi-defect. "
    "Enhancement is classical and flag-driven: gamma for exposure, percentile stretch for contrast, "
    "non-local means for noise, unsharp mask for blur; strength scales with the classifier's confidence."
)
