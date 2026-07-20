#!/usr/bin/env python3
"""FastAPI server serving the web frontend and prediction API."""

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from src.config import MODEL_PATH, ROOT_DIR
from src.predict import DepressionPredictor
from src.api_utils import result_to_json

app = FastAPI(title="Speech Depression Detection API")

FRONTEND_DIR = ROOT_DIR / "frontend"
predictor = None


def get_predictor():
    global predictor
    if predictor is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail="Model not trained. Run: python train.py",
            )
        predictor = DepressionPredictor()
    return predictor


@app.get("/api/health")
def health():
    model_ready = MODEL_PATH.exists()
    return {
        "status": "ok",
        "model_ready": model_ready,
        "message": "Ready" if model_ready else "Run python train.py first",
    }


@app.post("/api/predict")
async def predict_audio(
    file: UploadFile = File(...),
    chronic: bool = Form(False),
    recent_stress: bool = Form(False),
    postpartum: bool = Form(False),
    seasonal: bool = Form(False),
    mood_swings: bool = Form(False),
):
    allowed = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm"}
    suffix = Path(file.filename or "audio.wav").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported format. Use: {', '.join(allowed)}")

    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        pred = get_predictor()
        context = {
            "chronic": chronic,
            "recent_stress": recent_stress,
            "postpartum": postpartum,
            "seasonal": seasonal,
            "mood_swings": mood_swings,
        }
        result = pred.predict(tmp_path, context=context)
        return result_to_json(result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND_DIR / "static"), name="static")
