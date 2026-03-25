import json
from pathlib import Path

import librosa
import numpy as np

SR = 16000
N_MELS = 64
MAX_FRAMES = 300
TOP_DB = 20
EPS = 1e-9


def trim_and_normalize_waveform(y):
    y, _ = librosa.effects.trim(y, top_db=TOP_DB)
    if y.size == 0:
        return np.zeros(1, dtype=np.float32)

    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / (peak + EPS)

    return y.astype(np.float32)


def mel_from_waveform(y):
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_mels=N_MELS,
    )
    mel = librosa.power_to_db(mel, ref=np.max)

    if mel.shape[1] > MAX_FRAMES:
        mel = mel[:, :MAX_FRAMES]
    else:
        mel = np.pad(
            mel,
            ((0, 0), (0, MAX_FRAMES - mel.shape[1])),
            mode="constant",
        )

    return mel.astype(np.float32)


def extract_mel(audio_path):
    y, _ = librosa.load(audio_path, sr=SR, mono=True)
    y = trim_and_normalize_waveform(y)
    return mel_from_waveform(y)


def normalize_mel(mel):
    return ((mel - mel.mean()) / (mel.std() + EPS)).astype(np.float32)


def chunk_mel(mel, chunk_size=MAX_FRAMES, step=None):
    if step is None:
        step = chunk_size

    chunks = []
    total_frames = mel.shape[1]

    for start in range(0, total_frames, step):
        end = start + chunk_size
        chunk = mel[:, start:end]
        if chunk.shape[1] < chunk_size:
            pad = chunk_size - chunk.shape[1]
            chunk = np.pad(chunk, ((0, 0), (0, pad)), mode="constant")
        chunks.append(chunk.astype(np.float32))

        if end >= total_frames:
            break

    if not chunks:
        chunks.append(np.zeros((mel.shape[0], chunk_size), dtype=np.float32))

    return chunks


def load_threshold(config_path, default_threshold=0.5):
    config_file = Path(config_path)
    if not config_file.exists():
        return float(default_threshold)

    try:
        data = json.loads(config_file.read_text(encoding="utf-8"))
        return float(data.get("best_threshold", default_threshold))
    except Exception:
        return float(default_threshold)
