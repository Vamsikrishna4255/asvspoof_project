import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, roc_curve
from torch.utils.data import DataLoader

from dataset import MelDataset
from model_bal import StableCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-4

# project root and split paths
ROOT = Path(__file__).resolve().parents[1]
train_csv = ROOT / "splits" / "train_balanced.csv"
val_csv = ROOT / "splits" / "val.csv"
model_dir = ROOT / "models"
best_model_path = model_dir / "best_model.pth"
threshold_path = model_dir / "threshold.json"

train_dataset = MelDataset(str(train_csv))
val_dataset = MelDataset(str(val_csv))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

model = StableCNN().to(device)

train_counts = train_dataset.df["label"].value_counts().sort_index()
class_weights = torch.tensor(
    [
        len(train_dataset) / (2.0 * train_counts.get(0, 1)),
        len(train_dataset) / (2.0 * train_counts.get(1, 1)),
    ],
    dtype=torch.float32,
    device=device,
)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=LR)

model_dir.mkdir(exist_ok=True)
best_val_f1 = -1.0
best_threshold = 0.5


def evaluate_model(data_loader):
    model.eval()
    y_true = []
    y_prob = []
    total_loss = 0.0

    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            probs = torch.softmax(out, dim=1)[:, 1]

            total_loss += loss.item()
            y_true.extend(y.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    best_idx = np.argmax(tpr - fpr)
    threshold = float(thresholds[best_idx])
    y_pred = (y_prob >= threshold).astype(int)
    f1 = f1_score(y_true, y_pred)

    return {
        "loss": total_loss / max(len(data_loader), 1),
        "f1": float(f1),
        "threshold": threshold,
    }


for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    train_loss = total_loss / max(len(train_loader), 1)
    val_metrics = evaluate_model(val_loader)

    print(
        f"Epoch {epoch + 1} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_metrics['loss']:.4f} | "
        f"Val F1: {val_metrics['f1']:.4f} | "
        f"Threshold: {val_metrics['threshold']:.4f}"
    )

    if val_metrics["f1"] > best_val_f1:
        best_val_f1 = val_metrics["f1"]
        best_threshold = val_metrics["threshold"]
        torch.save(model.state_dict(), best_model_path)
        threshold_path.write_text(
            json.dumps(
                {
                    "best_threshold": best_threshold,
                    "best_val_f1": best_val_f1,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print("Saved improved checkpoint.")

print(f"Best validation F1: {best_val_f1:.4f}")
print(f"Best threshold saved: {best_threshold:.4f}")
