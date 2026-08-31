"""
Streamlit UI for the photo-quality classifier.

Upload a photo, get a per-defect probability read-out (blur, underexposure,
overexposure, noise, low contrast). The model tiles the image and aggregates,
so a defect anywhere in the frame is caught -- see src/predict.py.

Run locally:
    streamlit run app.py
Deploy: Streamlit Community Cloud, main file = app.py.
"""
import sys
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps

# The reusable inference code lives in src/ (same pattern as the notebooks).
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from predict import predict, DEFECT_COLUMNS, THRESHOLD, SHORT_SIDE  # noqa: E402

# Plain-language description shown next to each class.
BLURB = {
    "blur": "Out of focus or motion-smeared. Checked per-region, so a soft background counts.",
    "underexposed": "Too dark overall -- shot in too little light.",
    "overexposed": "Too bright -- highlights blown out to white.",
    "noise": "Grainy / speckled, the high-ISO look.",
    "contrast": "Flat and washed out -- tones bunched toward mid-grey.",
}

st.set_page_config(page_title="Photo Quality Classifier", page_icon="camera", layout="centered")

st.title("Photo Quality Classifier")
st.caption(
    "A from-scratch CNN that flags five kinds of photo defect at once. "
    "It slides a 256-pixel window across the whole image, scores each tile, and combines them "
    "(the maximum for blur, since blur can be local; the average for the rest, which are global)."
)

uploaded = st.file_uploader("Upload a photo", type=["png", "jpg", "jpeg", "webp"])

if uploaded is None:
    st.info("Pick an image to analyse. JPEG, PNG or WebP.")
    st.stop()

# exif_transpose fixes phone photos that carry a rotation flag instead of rotated pixels.
image = ImageOps.exif_transpose(Image.open(uploaded)).convert("RGB")

col_img, col_res = st.columns([1, 1], gap="large")

with col_img:
    st.image(image, caption=f"{image.size[0]} x {image.size[1]}", use_container_width=True)

with col_res:
    with st.spinner("Scanning..."):
        out = predict(image)

    flagged = [c for c in DEFECT_COLUMNS if out["flags"][c]]
    if flagged:
        st.error("Defects detected: " + ", ".join(flagged))
    else:
        st.success("No defects detected")
    st.caption(f"{out['n_tiles']} tiles scanned, short side resized to {SHORT_SIDE}px. "
               f"Threshold {THRESHOLD:.2f}.")

    for c in DEFECT_COLUMNS:
        p = out["probs"][c]
        mark = "**:red[FLAGGED]**" if out["flags"][c] else ":gray[clear]"
        st.markdown(f"**{c}** &nbsp; {p:.0%} &nbsp; {mark}")
        st.progress(min(max(p, 0.0), 1.0))
        st.caption(BLURB[c])

with st.expander("Per-tile detail"):
    st.caption(
        "Each column is one image tile; each row a defect. Values are the model's "
        "raw probability for that tile before aggregation."
    )
    rows = {c: [round(v, 2) for v in out["per_tile"][c]] for c in DEFECT_COLUMNS}
    st.dataframe(rows, use_container_width=True)

st.divider()
st.caption(
    "Model: iteration 4 (`models/traincombo_best.pt`), trained from scratch on synthetically "
    "degraded COCO photos. Test-set macro-F1 0.91 (single-defect) / 0.88 (multi-defect). "
    "The five classes above are the only defects it knows."
)
