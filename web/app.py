from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import shutil
import os
import sys
from pathlib import Path

# Add parent directory to path to import src
sys.path.insert(0, str(Path(__file__).parent.parent))

# prefer StableCNN-based inference
from src.inference_bal import predict_audio
from web.email_utils import send_spoof_alert

app = FastAPI()
templates = Jinja2Templates(directory="web/templates")

UPLOAD_DIR = "web/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # perform inference and normalize output for frontend
    raw = predict_audio(file_path)

    # convert keys to match what frontend expects
    result = {
        "prediction": raw.get("label"),
        "spoof_prob": raw.get("spoof_probability"),
        "real_prob": raw.get("real_probability"),
        # avg_spoof_prob is same as spoof_prob for now
        "avg_spoof_prob": raw.get("spoof_probability"),
        # include original raw values if needed
        **raw
    }

    # 🔴 Email alert ONLY for spoof
    if result.get("prediction") == "Spoof":
        send_spoof_alert(result.get("spoof_prob"))

    return result
