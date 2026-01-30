import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import MelDataset
from model import SimpleCNN

# ---------------- CONFIG ----------------
TEST_CSV = "splits/test.csv"
MODEL_PATH = "models/best_model.pth"
BATCH_SIZE = 16
# ---------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load dataset
test_loader = DataLoader(
    MelDataset(TEST_CSV),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# Load model
model = SimpleCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

y_true = []
y_pred = []
y_prob = []

# ---------------- EVALUATION ----------------
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        out = model(x)
        probs = torch.softmax(out, dim=1)

        preds = torch.argmax(probs, dim=1)

        y_true.extend(y.numpy())
        y_pred.extend(preds.cpu().numpy())
        y_prob.extend(probs[:, 1].cpu().numpy())  # spoof probability

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_prob = np.array(y_prob)

# ---------------- METRICS ----------------
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_prob)

print("\n📊 Evaluation Results")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC-AUC  : {auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["Real", "Spoof"]))

# ---------------- CONFUSION MATRIX ----------------
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(
    cm, annot=True, fmt="d",
    xticklabels=["Real", "Spoof"],
    yticklabels=["Real", "Spoof"]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
