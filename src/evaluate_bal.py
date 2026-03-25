from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, roc_curve

from audio_utils import load_threshold
from dataset import MelDataset
from model_bal import StableCNN

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
ROOT = Path(__file__).resolve().parents[1]
test_csv = ROOT / "splits" / "test.csv"
model_path = ROOT / "models" / "best_model.pth"
threshold_path = ROOT / "models" / "threshold.json"

# DataLoader
batch_size = 16
test_dataset = MelDataset(str(test_csv))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

# Model
model = StableCNN().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

all_probs = []
all_labels = []

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        out = model(x)
        probs = torch.softmax(out, dim=1)[:, 1]
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(y.numpy())

fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
best_idx = (tpr - fpr).argmax()
best_threshold = thresholds[best_idx]
saved_threshold = load_threshold(threshold_path, default_threshold=float(best_threshold))

print("Best threshold:", best_threshold)
print("Saved threshold:", saved_threshold)

preds = (np.array(all_probs) >= saved_threshold).astype(int)
print(classification_report(all_labels, preds, digits=4))
