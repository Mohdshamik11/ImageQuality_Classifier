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

## 3. Create a virtual environment

Open a terminal in VS Code (`` Ctrl+` ``) and run:

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You'll know it worked if you see `(venv)` at the start of your terminal prompt.

**Important:** every time you open a new terminal in VS Code for this project, re-activate the venv with the same command above. VS Code sometimes auto-detects and activates it for you — check for `(venv)` in the prompt to confirm.

## 4. Select the venv as your Python interpreter

Press `Ctrl+Shift+P` (`Cmd+Shift+P` on Mac) → type "Python: Select Interpreter" → choose the one that shows `./venv/...` in its path. This makes sure both your terminal and any notebooks use the same environment.

## 5. Install dependencies

With the venv active, run:

```bash
pip install -r requirements.txt
```

This installs PyTorch, OpenCV (for generating synthetic defects), Streamlit (for the UI), and everything else you need.

## 6. Verify it worked

Run this quick check in your terminal:

```bash
python -c "import torch; import cv2; import streamlit; print('All good, torch version:', torch.__version__)"
```

If that prints without errors, you're ready to go.

## 7. GPU check (optional)

If you have an NVIDIA GPU and want to use it for faster training:

```bash
python -c "import torch; print('GPU available:', torch.cuda.is_available())"
```

If this prints `False` and you do have an NVIDIA GPU, you likely need the CUDA-enabled build of PyTorch instead — see https://pytorch.org/get-started/locally/ for the correct install command for your system. If you don't have a GPU, that's fine — this project trains a small CNN, which runs on CPU without issue, just a bit slower per epoch.

---

Once this is done, you're ready to start the notebook for dataset generation and exploration.
