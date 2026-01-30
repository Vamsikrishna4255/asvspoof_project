import os
import librosa
import numpy as np
from tqdm import tqdm

# ---------------- CONFIG ----------------
BASE_DATA = r"C:\Users\krish\Downloads\asvspoof_project\data\ASVspoof2019_LA"
OUT_BASE  = r"C:\Users\krish\Downloads\asvspoof_project\mel_features"

SR = 16000
N_MELS = 64
MAX_FRAMES = 300
TOP_DB = 20
# ----------------------------------------


def extract_mel(audio_path):
    y, sr = librosa.load(audio_path, sr=SR, mono=True)
    y, _ = librosa.effects.trim(y, top_db=TOP_DB)

    # Normalize amplitude
    y = y / (np.max(np.abs(y)) + 1e-9)

    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_mels=N_MELS
    )
    mel = librosa.power_to_db(mel, ref=np.max)

    # Pad or truncate
    if mel.shape[1] > MAX_FRAMES:
        mel = mel[:, :MAX_FRAMES]
    else:
        mel = np.pad(
            mel,
            ((0, 0), (0, MAX_FRAMES - mel.shape[1])),
            mode="constant"
        )

    return mel


def process_split(split_name, out_dir):
    flac_dir = os.path.join(BASE_DATA, split_name, "flac")
    files = sorted(os.listdir(flac_dir))

    print(f"\nProcessing {split_name} | Total files: {len(files)}")

    for f in tqdm(files):
        in_path = os.path.join(flac_dir, f)
        out_path = os.path.join(out_dir, f.replace(".flac", ".npy"))

        if os.path.exists(out_path):
            continue  # skip already processed files

        mel = extract_mel(in_path)
        np.save(out_path, mel)


if __name__ == "__main__":
    process_split("ASVspoof2019_LA_train", os.path.join(OUT_BASE, "train"))
    process_split("ASVspoof2019_LA_dev",   os.path.join(OUT_BASE, "val"))
    process_split("ASVspoof2019_LA_eval",  os.path.join(OUT_BASE, "test"))

    print("\n✅ Preprocessing completed successfully")
