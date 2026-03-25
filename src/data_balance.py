import pandas as pd
from pathlib import Path

# project root
ROOT = Path(__file__).resolve().parents[1]
src_train = ROOT / "splits" / "train.csv"

df = pd.read_csv(src_train)

real_df = df[df["label"] == 0]
spoof_df = df[df["label"] == 1]

# Sample up to 10k each (don't exceed available samples)
sample_size = min(len(real_df), len(spoof_df), 10000)
real_df = real_df.sample(sample_size, random_state=42)
spoof_df = spoof_df.sample(sample_size, random_state=42)

balanced_df = pd.concat([real_df, spoof_df])
balanced_df = balanced_df.sample(frac=1).reset_index(drop=True)

balanced_df.to_csv(ROOT / "splits" / "train_balanced.csv", index=False)

print("Balanced distribution:")
print(balanced_df["label"].value_counts())