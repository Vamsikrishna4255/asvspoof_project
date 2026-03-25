import json

import numpy as np
import torch
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader

from dataset import MelDataset
from model_bal import StableCNN

# ---------------- CONFIG ----------------
TEST_CSV = "splits/test.csv"
MODEL_PATH = "models/best_model.pth"
THRESHOLD_PATH = "models/threshold.json"
BATCH_SIZE = 16
# --------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

test_loader = DataLoader(
    MelDataset(TEST_CSV),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

model = StableCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

y_true = []
y_spoof_prob = []

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)

        y_true.extend(y.numpy())
        y_spoof_prob.extend(probs[:, 1].cpu().numpy())

y_true = np.array(y_true)
y_spoof_prob = np.array(y_spoof_prob)

fpr, tpr, thresholds = roc_curve(y_true, y_spoof_prob)
youden_j = tpr - fpr
best_idx = np.argmax(youden_j)
best_threshold = thresholds[best_idx]

print("Auto-found best threshold")
print("Best threshold:", round(float(best_threshold), 4))
print("TPR (Recall):", round(float(tpr[best_idx]), 4))
print("FPR:", round(float(fpr[best_idx]), 4))

with open(THRESHOLD_PATH, "w", encoding="utf-8") as f:
    json.dump(
        {
            "best_threshold": float(best_threshold),
            "recall": float(tpr[best_idx]),
            "fpr": float(fpr[best_idx]),
        },
        f,
        indent=2,
    )

print("Saved threshold config to", THRESHOLD_PATH)
