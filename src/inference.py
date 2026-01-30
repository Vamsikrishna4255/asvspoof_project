import numpy as np
import librosa
import torch
from .model import SimpleCNN

# ---------------- CONFIG ----------------
MODEL_PATH = "models/best_model.pth"

SR = 16000
N_MELS = 64
MAX_FRAMES = 300
TOP_DB = 20

CHUNK_SECONDS = 3.0          # segment length
SPOOF_THRESHOLD = 0.45       # CRITICAL FIX
# --------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = SimpleCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()


def extract_mel_from_wave(y):
    """Exact same preprocessing as training"""
    # silence trim
    y, _ = librosa.effects.trim(y, top_db=TOP_DB)

    # normalize waveform
    y = y / (np.max(np.abs(y)) + 1e-9)

    # mel-spectrogram
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_mels=N_MELS
    )
    mel = librosa.power_to_db(mel, ref=np.max)

    # pad / truncate
    if mel.shape[1] > MAX_FRAMES:
        mel = mel[:, :MAX_FRAMES]
    else:
        mel = np.pad(
            mel,
            ((0, 0), (0, MAX_FRAMES - mel.shape[1])),
            mode="constant"
        )

    # normalize mel
    mel = (mel - mel.mean()) / (mel.std() + 1e-9)
    return mel


def predict_audio(audio_path):
    y, _ = librosa.load(audio_path, sr=SR, mono=True)

    chunk_len = int(SR * CHUNK_SECONDS)
    spoof_scores = []

    if len(y) < chunk_len:
        mel = extract_mel_from_wave(y)
        x = torch.tensor(mel).unsqueeze(0).unsqueeze(0).float().to(device)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
        spoof_scores.append(probs[1])
    else:
        for i in range(0, len(y) - chunk_len + 1, chunk_len):
            chunk = y[i:i + chunk_len]
            mel = extract_mel_from_wave(chunk)
            x = torch.tensor(mel).unsqueeze(0).unsqueeze(0).float().to(device)

            with torch.no_grad():
                probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()

            spoof_scores.append(probs[1])

    avg_spoof = float(np.mean(spoof_scores))
    avg_real = 1.0 - avg_spoof

    if avg_spoof >= 0.53:
        label = "Real"
    else:
        label = "Spoof"

    return {
        "prediction": label,
        "avg_spoof_prob": round(avg_spoof, 4),
        "avg_real_prob": round(avg_real, 4),
        "num_segments": len(spoof_scores)
    }


# ---------------- RUN EXAMPLE ----------------
if __name__ == "__main__":
    # CHANGE THIS PATH TO YOUR AUDIO FILE
    AUDIO_PATH = r"C:\Users\krish\Downloads\asvspoof_project\test_audios\test-audio1.wav"

    result = predict_audio(AUDIO_PATH)
    print(result)
