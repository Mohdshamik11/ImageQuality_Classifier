# Setup Guide

Follow these steps in order before writing any project code.

## 1. VS Code extensions

Open VS Code, go to the Extensions panel (`Ctrl+Shift+X` / `Cmd+Shift+X`), and install:

- **Python** (by Microsoft) — core language support, linting, debugging
- **Jupyter** (by Microsoft) — lets you create and run `.ipynb` notebooks directly in VS Code
- **Pylance** (by Microsoft) — usually installs automatically with Python; gives you better autocomplete

That's all you need. No other extensions are required for this project.

## 2. Open the project folder

Open this `photo-quality-classifier/` folder in VS Code: `File > Open Folder...`

## 3. Install Miniconda (if you don't have conda already)

Download from https://docs.conda.io/en/latest/miniconda.html, pick the installer for your OS, and accept the defaults during install. Restart your terminal afterward.

Verify it worked:
```bash
conda --version
```

## 4. Create and activate the conda environment

Open a terminal in VS Code (`` Ctrl+` ``) and run:

```bash
conda create -n photo-quality-classifier python=3.11
conda activate photo-quality-classifier
```

You'll know it worked if you see `(photo-quality-classifier)` at the start of your terminal prompt.

**Important:** every time you open a new terminal in VS Code for this project, re-run `conda activate photo-quality-classifier`.

**If you have an NVIDIA GPU** and want CUDA support, install PyTorch through conda first, before the pip install below:
```bash
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
```

## 5. Select the conda environment as your Python interpreter

Press `Ctrl+Shift+P` (`Cmd+Shift+P` on Mac) → type "Python: Select Interpreter" → choose the one labeled `photo-quality-classifier (conda)`. This makes sure both your terminal and any notebooks use the same environment.

## 6. Install dependencies

With the environment active, run:

```bash
pip install -r requirements-dev.txt
```

This installs PyTorch, OpenCV (for generating synthetic defects), Streamlit (for the UI), pandas / scikit-learn / matplotlib (data pipeline + notebooks), and everything else you need.

(`requirements.txt` is the trimmed, app-only list that Streamlit Community Cloud installs when the app is deployed. `requirements-dev.txt` includes it plus the pipeline and notebook tools.)

## 7. Verify it worked

Run this quick check in your terminal:

```bash
python -c "import torch; import cv2; import streamlit; print('All good, torch version:', torch.__version__)"
```

If that prints without errors, you're ready to go.

## 8. GPU check (optional)

If you have an NVIDIA GPU and want to use it for faster training:

```bash
python -c "import torch; print('GPU available:', torch.cuda.is_available())"
```

If this prints `False` and you do have an NVIDIA GPU, revisit step 4 — you likely need to (re)install the CUDA-enabled PyTorch build via conda before installing the rest of `requirements.txt`. If you don't have a GPU, that's fine — this project trains a small CNN, which runs on CPU without issue, just a bit slower per epoch.

---

Once this is done, you're ready to start the notebook for dataset generation and exploration.