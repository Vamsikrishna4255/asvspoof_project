import torch
import numpy as np
import matplotlib.pyplot as plt

from model_bal import StableCNN

# ---------------- CONFIG ----------------
MODEL_PATH = "models/best_model.pth"
SAMPLE_MEL = "mel_features/test/LA_E_1000147.npy"  # change if needed
TARGET_CLASS = 1  # 0 = real, 1 = spoof
# ---------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = StableCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Load mel-spectrogram
mel = np.load(SAMPLE_MEL)
mel_norm = (mel - mel.mean()) / (mel.std() + 1e-9)

x = torch.tensor(mel_norm).unsqueeze(0).unsqueeze(0).float().to(device)

# ---------------- GRAD-CAM ----------------
gradients = []
activations = []

def forward_hook(module, input, output):
    activations.append(output)

def backward_hook(module, grad_input, grad_output):
    gradients.append(grad_output[0])

# Register hooks on last conv layer
target_layer = model.features[-2]
target_layer.register_forward_hook(forward_hook)
target_layer.register_backward_hook(backward_hook)

# Forward pass
out = model(x)
pred_class = out.argmax(dim=1).item()

# Backward pass
model.zero_grad()
out[:, TARGET_CLASS].backward()

# Compute Grad-CAM
grads = gradients[0]
acts = activations[0]

weights = grads.mean(dim=(2, 3), keepdim=True)
cam = (weights * acts).sum(dim=1).squeeze().detach().cpu().numpy()

cam = np.maximum(cam, 0)
cam = cam / cam.max()

# ---------------- VISUALIZATION ----------------
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.title("Original Mel-Spectrogram")
plt.imshow(mel, aspect="auto", origin="lower", cmap="magma")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.title("Grad-CAM Heatmap")
plt.imshow(cam, aspect="auto", origin="lower", cmap="jet")
plt.colorbar()

plt.tight_layout()
plt.show()

print("Predicted class:", "Spoof" if pred_class == 1 else "Real")
