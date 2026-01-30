import os
import pandas as pd

# ---------------- CONFIG ----------------
BASE_DATA = r"C:\Users\krish\Downloads\asvspoof_project\data\ASVspoof2019_LA"
MEL_BASE  = r"C:\Users\krish\Downloads\asvspoof_project\mel_features"
OUT_DIR   = r"C:\Users\krish\Downloads\asvspoof_project\splits"

LABEL_MAP = {"bonafide": 0, "spoof": 1}
# ----------------------------------------


def read_protocol(protocol_path):
    rows = []
    with open(protocol_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            file_id = parts[1]
            label = LABEL_MAP[parts[-1]]
            rows.append((file_id, label))
    return rows


def build_csv(protocol_file, mel_dir, out_csv):
    entries = read_protocol(protocol_file)
    data = []

    for file_id, label in entries:
        mel_path = os.path.join(mel_dir, file_id + ".npy")
        if os.path.exists(mel_path):
            data.append([mel_path, label])

    df = pd.DataFrame(data, columns=["path", "label"])
    df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv} | Samples: {len(df)}")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    build_csv(
        protocol_file=os.path.join(
            BASE_DATA, "ASVspoof2019_LA_cm_protocols",
            "ASVspoof2019.LA.cm.train.trn.txt"
        ),
        mel_dir=os.path.join(MEL_BASE, "train"),
        out_csv=os.path.join(OUT_DIR, "train.csv")
    )

    build_csv(
        protocol_file=os.path.join(
            BASE_DATA, "ASVspoof2019_LA_cm_protocols",
            "ASVspoof2019.LA.cm.dev.trl.txt"
        ),
        mel_dir=os.path.join(MEL_BASE, "val"),
        out_csv=os.path.join(OUT_DIR, "val.csv")
    )

    build_csv(
        protocol_file=os.path.join(
            BASE_DATA, "ASVspoof2019_LA_cm_protocols",
            "ASVspoof2019.LA.cm.eval.trl.txt"
        ),
        mel_dir=os.path.join(MEL_BASE, "test"),
        out_csv=os.path.join(OUT_DIR, "test.csv")
    )

    print("\n✅ CSV split generation completed")
