import numpy as np
import pandas as pd
import os
import json
from sklearn.metrics import confusion_matrix, classification_report
from scripts.utils import get_project_path

def load_model_outputs(model_name, base_dir = get_project_path("models")):
    model_dir = os.path.join(base_dir, model_name)
    y_true = np.load(os.path.join(model_dir, "y_true.npy"))
    recon_error = np.load(os.path.join(model_dir, "recon_error.npy"))
    with open(os.path.join(model_dir, "metrics.json")) as f:
        metrics = json.load(f)

    threshold = metrics["best_thresh"]
    y_pred = recon_error >= threshold
    return y_true.astype(bool), y_pred.astype(bool), metrics

def analyze_model(name, y_true, y_pred, metrics):
    print(f"-- {name} -- ")
    print(f"F1 Score: {metrics['best_f1']:.4f} | ROC AUC: {metrics['roc_auc']:.4f} | PR AUC: {metrics['pr_auc']:.4f}")
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"Confusion Matrix:\nTP={tp}, FP = {fp}, FN = {fn}, TN = {tn}")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Abnormal']))
    print()


