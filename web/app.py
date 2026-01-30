from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import shutil
import os

from src.inference import predict_audio
from .email_utils import send_spoof_alert

app = FastAPI()
templates = Jinja2Templates(directory="web/templates")

UPLOAD_DIR = "web/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict_audio_api(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = predict_audio(file_path)

    # 🚨 EMAIL ALERT ONLY IF SPOOF
    if result["prediction"] == "Spoof":
        send_spoof_alert(result["avg_spoof_prob"])

    return result
