"""
Evaluation metrics for the multi-label photo-quality classifier.

Two things live here:
  evaluate_model()  - run a whole DataLoader through a model, return the true
                      labels, the predicted probabilities, and the average loss.
  compute_metrics() - turn (true labels, predicted probs) into per-class and
                      macro numbers.

Metric choices (see the project plan):
  - per-class precision / recall / F1
  - macro-F1  = plain average of the 5 per-class F1s  <- the headline number
  - per-class PR-AUC (area under precision-recall curve; threshold-independent)
  - raw per-class TP / FP / FN / TN
  - subset accuracy = fraction of images with ALL 5 labels correct
  - plain accuracy and micro-F1 are deliberately NOT reported (misleading here)

Threshold is 0.5 for every class in the baseline.

New tools:
  sklearn.metrics              - tested implementations of the scores above.
  model.eval() / torch.no_grad - evaluation mode + no gradient tracking, for the
                                 measure-only passes (faster, less memory).
  torch.sigmoid                - turns raw model scores (logits) into 0-1 probs.

Usage (sanity check on random data):
    python src/metrics.py
"""
import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    multilabel_confusion_matrix,
    precision_score,
    recall_score,
)

DEFECT_COLUMNS = ["blur", "underexposed", "overexposed", "noise", "contrast"]


@torch.no_grad()  # nothing in here should build a gradient graph
def evaluate_model(model, loader, device, criterion=None):
    """Push every batch in `loader` through `model`.

    Returns:
      y_true : (N, 5) numpy array of 0/1 ground-truth labels
      y_prob : (N, 5) numpy array of predicted probabilities in [0, 1]
      avg_loss : float if `criterion` was given, else None
    Row order matches the loader's dataset order (val/test loaders are unshuffled,
    so it also matches loader.dataset.df row-for-row).
    """
    model.eval()  # switch off training-only behaviour (e.g. dropout, if added later)

    true_chunks, prob_chunks = [], []
    total_loss, n_batches = 0.0, 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)  # (B, 5) raw scores

        if criterion is not None:
            total_loss += criterion(logits, labels).item()
            n_batches += 1

        probs = torch.sigmoid(logits)  # (B, 5) -> [0, 1]
        true_chunks.append(labels.cpu().numpy())
        prob_chunks.append(probs.cpu().numpy())

    y_true = np.concatenate(true_chunks)
    y_prob = np.concatenate(prob_chunks)
    avg_loss = (total_loss / n_batches) if criterion is not None else None
    return y_true, y_prob, avg_loss


def compute_metrics(y_true, y_prob, threshold=0.5, class_names=DEFECT_COLUMNS):
    """(N, 5) true labels + (N, 5) predicted probabilities -> nested dict of metrics.

    threshold: a single number applied to every class, OR an array-like of length
    len(class_names) giving a per-class cutoff (as produced by tune_thresholds()).
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    threshold = np.asarray(threshold, dtype=float)     # scalar (0-d) or shape (n_classes,)
    y_pred = (y_prob >= threshold).astype(int)         # broadcasts (N,5) >= (5,) -> (N,5)

    # (n_classes, 2, 2): each class's [[TN, FP], [FN, TP]]
    mcm = multilabel_confusion_matrix(y_true, y_pred, labels=range(len(class_names)))

    per_class = {}
    for i, name in enumerate(class_names):
        tn, fp, fn, tp = mcm[i].ravel()

        # PR-AUC is only defined when the class has both a positive and a
        # negative example in y_true (true for the combo subset if a class
        # happens to be absent there).
        if y_true[:, i].min() != y_true[:, i].max():
            pr_auc = float(average_precision_score(y_true[:, i], y_prob[:, i]))
        else:
            pr_auc = float("nan")

        per_class[name] = {
            # zero_division=0: if the model predicted no positives for a class,
            # call precision/recall 0 instead of raising a warning.
            "precision": float(precision_score(y_true[:, i], y_pred[:, i], zero_division=0)),
            "recall": float(recall_score(y_true[:, i], y_pred[:, i], zero_division=0)),
            "f1": float(f1_score(y_true[:, i], y_pred[:, i], zero_division=0)),
            "pr_auc": pr_auc,
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        }

    f1s = [per_class[n]["f1"] for n in class_names]
    precs = [per_class[n]["precision"] for n in class_names]
    recs = [per_class[n]["recall"] for n in class_names]

    return {
        "macro_f1": float(np.mean(f1s)),               # <- headline number
        "macro_precision": float(np.mean(precs)),
        "macro_recall": float(np.mean(recs)),
        "subset_accuracy": float((y_pred == y_true).all(axis=1).mean()),
        "per_class": per_class,
    }


def tune_thresholds(y_true, y_prob, grid=None, class_names=DEFECT_COLUMNS):
    """For each class independently, pick the threshold in `grid` that maximises
    that class's F1 on (y_true, y_prob). Returns a numpy array of per-class
    thresholds, same length/order as class_names.

    Run this on VALIDATION data only. Feed the result straight into
    compute_metrics(..., threshold=<this array>) and keep it fixed for the test set.
    """
    if grid is None:
        grid = np.round(np.arange(0.05, 0.96, 0.05), 2)  # 0.05, 0.10, ..., 0.95

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    best = np.full(len(class_names), 0.5)

    for i in range(len(class_names)):
        best_f1 = -1.0
        for t in grid:
            pred = (y_prob[:, i] >= t).astype(int)
            f1 = f1_score(y_true[:, i], pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best[i] = float(t)
    return best


def combo_mask(filenames):
    """Boolean numpy array: True where the filename is a combo (multi-defect) image."""
    return np.array(["_combo_" in str(f) for f in filenames])


def print_metrics(metrics, title="metrics"):
    """Pretty-print the dict from compute_metrics."""
    print(f"\n=== {title} ===")
    print(f"macro-F1 {metrics['macro_f1']:.3f} | "
          f"macro-P {metrics['macro_precision']:.3f} | "
          f"macro-R {metrics['macro_recall']:.3f} | "
          f"subset-acc {metrics['subset_accuracy']:.3f}")
    print(f"{'class':<14}{'prec':>7}{'recall':>8}{'f1':>7}{'pr_auc':>8}"
          f"{'tp':>6}{'fp':>6}{'fn':>6}{'tn':>6}")
    for name, m in metrics["per_class"].items():
        print(f"{name:<14}{m['precision']:>7.3f}{m['recall']:>8.3f}{m['f1']:>7.3f}"
              f"{m['pr_auc']:>8.3f}{m['tp']:>6}{m['fp']:>6}{m['fn']:>6}{m['tn']:>6}")


if __name__ == "__main__":
    # Plumbing check only: random labels + random probabilities. Numbers will be
    # meaningless (~0.5 F1); we just want no crashes and a sane-looking table.
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=(300, 5))
    y_prob = rng.random(size=(300, 5))

    print_metrics(compute_metrics(y_true, y_prob), "random-data sanity check (threshold 0.5)")

    thr = tune_thresholds(y_true, y_prob)
    print("\ntuned thresholds (random data, so meaningless):",
          dict(zip(DEFECT_COLUMNS, thr.round(2))))
    print_metrics(compute_metrics(y_true, y_prob, threshold=thr),
                  "random-data, per-class tuned thresholds")

    mask = combo_mask(["raw_0001_blur.png", "raw_0002_combo_blur_noise.png"])
    print("\ncombo_mask example:", mask)
