import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve
from dataset import MelDataset
from model import SimpleCNN

# ---------------- CONFIG ----------------
TEST_CSV = "splits/test.csv"
MODEL_PATH = "models/best_model.pth"
BATCH_SIZE = 16
# --------------------------------------

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
y_spoof_prob = []

# ---------------- COLLECT PROBABILITIES ----------------
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)

        y_true.extend(y.numpy())
        y_spoof_prob.extend(probs[:, 1].cpu().numpy())

y_true = np.array(y_true)
y_spoof_prob = np.array(y_spoof_prob)

# ---------------- FIND BEST THRESHOLD ----------------
fpr, tpr, thresholds = roc_curve(y_true, y_spoof_prob)

youden_j = tpr - fpr
best_idx = np.argmax(youden_j)
best_threshold = thresholds[best_idx]

print("\n✅ AUTO-FOUND BEST THRESHOLD")
print("Best threshold:", round(float(best_threshold), 4))
print("TPR (Recall):", round(float(tpr[best_idx]), 4))
print("FPR:", round(float(fpr[best_idx]), 4))
