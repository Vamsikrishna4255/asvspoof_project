import random

import numpy as np
import torch

from .audio_utils import (
    MAX_FRAMES,
    chunk_mel,
    extract_mel,
    load_threshold,
    normalize_mel,
)
from .model_bal import StableCNN

# ==============================
# DETERMINISTIC SETTINGS
# ==============================

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ==============================
# CONFIG
# ==============================

MODEL_PATH = "models/best_model.pth"
THRESHOLD_CONFIG_PATH = "models/threshold.json"
BEST_THRESHOLD = load_threshold(THRESHOLD_CONFIG_PATH, default_threshold=0.50)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# LOAD MODEL
# ==============================

model = StableCNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

for param in model.parameters():
    param.requires_grad = False

print("Model loaded successfully.")


# ==============================
# FULLY STABLE PREDICTION
# ==============================

def predict_audio(audio_path):
    mel = normalize_mel(extract_mel(audio_path))

    probs_list = []

    for chunk in chunk_mel(mel, chunk_size=MAX_FRAMES, step=MAX_FRAMES):
        x = torch.tensor(chunk).unsqueeze(0).unsqueeze(0).float().to(DEVICE)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        probs = np.nan_to_num(probs)
        probs_list.append(probs)

    stacked_probs = np.stack(probs_list, axis=0)
    avg_probs = np.mean(stacked_probs, axis=0)
    max_probs = np.max(stacked_probs, axis=0)

    real_prob = float(avg_probs[0])
    spoof_prob = float(0.5 * avg_probs[1] + 0.5 * max_probs[1])

    real_prob = max(0.0, min(1.0, real_prob))
    spoof_prob = max(0.0, min(1.0, spoof_prob))

    label = "Spoof" if spoof_prob >= BEST_THRESHOLD else "Real"

    return {
        "label": label,
        "real_probability": round(real_prob, 4),
        "spoof_probability": round(spoof_prob, 4),
        "threshold": round(BEST_THRESHOLD, 4),
        "avg_spoof_probability": round(float(avg_probs[1]), 4),
        "max_spoof_probability": round(float(max_probs[1]), 4),
    }


if __name__ == "__main__":
    test_path = r"C:\Users\krish\Downloads\asvspoof_project\test_audios\test-audio(R)-1.wav"
    result = predict_audio(test_path)
    print(result)
