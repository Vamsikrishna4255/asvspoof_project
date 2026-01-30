import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from dataset import MelDataset
from model import SimpleCNN

# ---------------- CONFIG ----------------
TRAIN_CSV = "splits/train.csv"
VAL_CSV   = "splits/val.csv"
MODEL_DIR = "models"

BATCH_SIZE = 16
EPOCHS = 3
LR = 1e-4
# ---------------------------------------

os.makedirs(MODEL_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Data loaders
train_loader = DataLoader(
    MelDataset(TRAIN_CSV),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    MelDataset(VAL_CSV),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# Model
model = SimpleCNN().to(device)

# ⚠️ CLASS-WEIGHTED LOSS (CRITICAL)
num_real = 4098
num_spoof = 36081

class_weights = torch.tensor(
    [1.0 / num_real, 1.0 / num_spoof],
    device=device
)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ---------------- TRAINING ----------------
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for x, y in loop:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} | Avg Train Loss: {avg_loss:.4f}")

    # Save checkpoint
    torch.save(
        model.state_dict(),
        os.path.join(MODEL_DIR, "best_model.pth")
    )

print("\n✅ Training completed successfully")
