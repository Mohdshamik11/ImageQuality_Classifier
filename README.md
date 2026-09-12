# Photo Quality Classifier

A convolutional neural network, **trained from scratch**, that inspects a photograph and flags
five kinds of quality defect at once — **blur, underexposure, overexposure, sensor noise, and
low contrast** — then enhances the photo to fix them.

**Live demo:** https://imagequalityclassifier.streamlit.app
**Full build log:** [`docs/writeup.html`](docs/writeup.html) — data pipeline, model training, and
the reasoning behind every decision.

---

## What it does

- **Multi-label classification.** A photo can be blurry *and* underexposed *and* noisy at once,
  so the model has five independent yes/no outputs, not one "pick a class."
- **Tiled inference.** Uploads are scanned by sliding a 256-pixel window across the whole frame,
  so a defect anywhere in the image is caught, not just ones in the middle of the frame.
- **Enhancement.** Exposure, contrast, and noise are fixed with classical image processing,
  scaled to how severe the classifier judges each defect to be. Blur is fixed by a custom-trained
  GAN (generative adversarial network) — a small U-Net trained specifically to deblur, since a
  plain sharpening filter can only amplify edges that already exist, not recover detail that's
  genuinely been lost.

---

## How it works

1. Upload up to 15 photos.
2. The classifier scans each one and flags which of the five defects are present.
3. Adjust the **enhancement strength** slider to control how much correction gets applied.
4. Click **Enhance** — flagged defects are corrected and you get a before/after comparison.

---

## Running it

**Requires:** [conda](https://docs.conda.io/en/latest/miniconda.html). See [`SETUP.md`](SETUP.md)
for the full first-time setup.

```bash
git clone https://github.com/Mohdshamik11/ImageQuality_Classifier.git
cd ImageQuality_Classifier

conda create -n imageQuality_Classifier python=3.11
conda activate imageQuality_Classifier
pip install -r requirements-dev.txt        # full env; requirements.txt alone is app-only
```

Trained models ship with the repo, so the app runs immediately:

```bash
streamlit run app.py
```

---

## Project structure

```
📄 app.py            Streamlit UI — the deployment entry point
📁 src/               classifier, enhancement, and training code
📁 notebooks/         classifier training notebooks
📁 models/            trained model weights
📁 docs/              full build log (writeup.html)
```

---

## Known limits

- **Severely blurred or blown-out photos** can't be fully recovered — once detail is truly gone
  (heavy motion blur, pixels clipped to pure white/black), no amount of correction can bring it
  back; enhancement can only work with what's actually left in the pixels.
- **Very large photos** are deblurred at a capped working resolution, then scaled back up — so
  blur correction on a big photo ends up slightly softer than on a smaller one.
- **Intentional soft focus** (e.g. portrait bokeh) isn't distinguished from a real blur defect —
  the model wasn't trained to tell the two apart.
- **No dedicated face-restoration step** — faces in heavily degraded photos improve less
  reliably than the rest of the frame.

---

## Notes

- Built as a learning project — the emphasis was on understanding each step, not just the final
  number. `docs/writeup.html` walks through the reasoning.
- Deployed on Streamlit Community Cloud's free tier (CPU, ~1 GB RAM); the 15-image cap and model
  sizes are chosen for it.
