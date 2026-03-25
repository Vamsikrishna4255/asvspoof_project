import os
from tqdm import tqdm
from audio_utils import extract_mel

# ---------------- CONFIG ----------------
BASE_DATA = r"C:\Users\krish\Downloads\asvspoof_project\data\ASVspoof2019_LA"
OUT_BASE  = r"C:\Users\krish\Downloads\asvspoof_project\mel_features"

# ----------------------------------------


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
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.save(out_path, mel)


if __name__ == "__main__":
    process_split("ASVspoof2019_LA_train", os.path.join(OUT_BASE, "train"))
    process_split("ASVspoof2019_LA_dev",   os.path.join(OUT_BASE, "val"))
    process_split("ASVspoof2019_LA_eval",  os.path.join(OUT_BASE, "test"))

    print("\n✅ Preprocessing completed successfully")
