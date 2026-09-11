#!/usr/bin/env bash
# Sets up and launches the blur-specialist GAN fine-tune on a RunPod pod.
#
# Run this INSIDE the pod (SSH, or the Jupyter terminal). Before running it,
# scp the shrunk clean pool to the pod's persistent volume once:
#   scp -P <port> -r data/clean_pool_small root@<pod-ip>:/workspace/data/
#
# Usage:
#   bash runpod_train.sh                       # defaults: 25 epochs, batch 16
#   EPOCHS=30 BATCH_SIZE=24 bash runpod_train.sh
#
# Getting this script onto a fresh pod (pick one):
#   - paste its contents into `nano runpod_train.sh`, or
#   - once pushed to GitHub:
#     curl -o runpod_train.sh https://raw.githubusercontent.com/Mohdshamik11/ImageQuality_Classifier/restore-gan/runpod_train.sh
set -euo pipefail

REPO_URL="https://github.com/Mohdshamik11/ImageQuality_Classifier.git"
BRANCH="restore-gan"
WORKDIR="/workspace"
REPO_DIR="$WORKDIR/ImageQuality_Classifier"
DATA_DIR="$WORKDIR/data/clean_pool_small"

EPOCHS="${EPOCHS:-25}"
BATCH_SIZE="${BATCH_SIZE:-16}"
WORKERS="${WORKERS:-4}"

echo "=== 1. code ==="
if [ -d "$REPO_DIR/.git" ]; then
    echo "repo already present -- pulling latest $BRANCH"
    cd "$REPO_DIR"
    git fetch origin "$BRANCH"
    git checkout "$BRANCH"
    git reset --hard "origin/$BRANCH"      # discards any local edits in the pod's copy
else
    git clone -b "$BRANCH" "$REPO_URL" "$REPO_DIR"
    cd "$REPO_DIR"
fi

echo
echo "=== 2. data check ==="
if [ ! -d "$DATA_DIR/coco" ] || [ ! -d "$DATA_DIR/div2k" ]; then
    echo "ERROR: expected $DATA_DIR/coco and $DATA_DIR/div2k."
    echo "       scp the clean pool to the pod first:"
    echo "       scp -P <port> -r data/clean_pool_small root@<pod-ip>:/workspace/data/"
    exit 1
fi
echo "coco:  $(find "$DATA_DIR/coco" -type f | wc -l) files"
echo "div2k: $(find "$DATA_DIR/div2k" -type f | wc -l) files"

echo
echo "=== 3. dependencies ==="
pip install -q -r requirements-dev.txt

echo
echo "=== 4. GPU check ==="
python -c "
import torch
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device:', torch.cuda.get_device_name(0))
    print('VRAM  :', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), 'GB')
"

echo
echo "=== 5. training (tmux session 'train') ==="
if tmux has-session -t train 2>/dev/null; then
    echo "a tmux session named 'train' already exists."
    echo "  watch it   : tmux attach -t train   (Ctrl+B then D to detach again)"
    echo "  or         : tail -f $WORKDIR/train.log"
    exit 0
fi

CMD="cd $REPO_DIR && python src/train_restore_gan.py \
    --clean-dirs $DATA_DIR/coco $DATA_DIR/div2k \
    --out-dir $WORKDIR \
    --epochs $EPOCHS --batch-size $BATCH_SIZE --workers $WORKERS \
    2>&1 | tee $WORKDIR/train.log"

tmux new -d -s train "$CMD"
echo "started in tmux session 'train' ($EPOCHS epochs, batch $BATCH_SIZE)."
echo "  watch live : tmux attach -t train   (Ctrl+B then D to detach again)"
echo "  or         : tail -f $WORKDIR/train.log"
echo
echo "when it's done, pull results back to your PC:"
echo "  scp -P <port> root@<pod-ip>:$WORKDIR/models/restore_blur_gan*.pt models/"
echo "  scp -P <port> root@<pod-ip>:$WORKDIR/models/restore_gan_history.csv models/"
echo "  scp -P <port> -r root@<pod-ip>:$WORKDIR/outputs/restore_gan_samples ./outputs/"
echo
echo "then STOP or TERMINATE the pod -- billing continues while it's running."
