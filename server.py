#!/usr/bin/env python3
"""FastAPI server serving the web frontend and prediction API."""

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from src.config import (
    ACOUSTIC_MODEL_PATH,
    ACTIVE_MODEL,
    MODEL_PATH,
    ROOT_DIR,
    SSL_MODEL_PATH,
)
from src.predict import DepressionPredictor
from src.api_utils import result_to_json
from src.patient_store import (
    save_patient_record,
    list_records,
    get_record,
    get_patient_history,
    find_patient,
    resolve_patient_identity,
    record_to_view,
    delete_record,
    delete_patient,
)

app = FastAPI(title="Speech Depression Detection API")

FRONTEND_DIR = ROOT_DIR / "frontend"
predictor = None


def _model_ready() -> bool:
    if ACTIVE_MODEL == "acoustic":
        return ACOUSTIC_MODEL_PATH.exists()
    if ACTIVE_MODEL == "ssl":
        return SSL_MODEL_PATH.exists()
    return MODEL_PATH.exists()


def get_predictor():
    global predictor
    if predictor is None:
        if not _model_ready():
            raise HTTPException(
                status_code=503,
                detail="Model not trained. Run: python train_ssl.py",
            )
        predictor = DepressionPredictor()
    return predictor


@app.get("/api/health")
def health():
    acoustic_ready = ACOUSTIC_MODEL_PATH.exists()
    ssl_ready = SSL_MODEL_PATH.exists()
    cnn_ready = MODEL_PATH.exists()
    model_ready = _model_ready()
    return {
        "status": "ok",
        "model_ready": model_ready,
        "active_model": ACTIVE_MODEL if model_ready else None,
        "acoustic_ready": acoustic_ready,
        "ssl_ready": ssl_ready,
        "cnn_ready": cnn_ready,
        "message": (
            "Ready"
            if model_ready
            else "Run python train_official_acoustic.py first"
        ),
    }


@app.post("/api/predict")
async def predict_audio(
    file: UploadFile = File(...),
    chronic: bool = Form(False),
    recent_stress: bool = Form(False),
    postpartum: bool = Form(False),
    seasonal: bool = Form(False),
    mood_swings: bool = Form(False),
    name: str = Form(""),
    age: str = Form(""),
    id_number: str = Form(""),
    patient_id: str = Form(""),
    gender: str = Form(""),
    phone: str = Form(""),
    notes: str = Form(""),
    save_record: bool = Form(True),
):
    allowed = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm"}
    suffix = Path(file.filename or "audio.wav").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported format. Use: {', '.join(allowed)}")

    if not (name or "").strip():
        raise HTTPException(status_code=400, detail="Patient name is required")
    if not (patient_id or "").strip() and not (id_number or "").strip():
        raise HTTPException(status_code=400, detail="Patient ID or ID number is required")

    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    age_val = None
    if (age or "").strip():
        try:
            age_val = int(float(age))
            if age_val < 1 or age_val > 120:
                raise ValueError
        except ValueError:
            raise HTTPException(status_code=400, detail="Age must be a number between 1 and 120")

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
            "patient_id": patient_id,
            "id_number": id_number,
            "participant_id": patient_id or id_number,
            "original_filename": file.filename or "",
        }
        result = pred.predict(tmp_path, context=context)
        payload = result_to_json(result)

        patient = resolve_patient_identity({
            "name": name,
            "age": age_val,
            "id_number": id_number,
            "patient_id": patient_id,
            "gender": gender,
            "phone": phone,
            "notes": notes,
        })
        payload["patient"] = {
            "name": patient.get("name") or "",
            "age": patient.get("age"),
            "id_number": patient.get("id_number") or "",
            "patient_id": patient.get("patient_id") or "",
            "gender": patient.get("gender") or "",
            "phone": patient.get("phone") or "",
            "notes": patient.get("notes") or "",
        }

        saved = None
        if save_record:
            saved = save_patient_record(
                patient=patient,
                analysis=payload,
                audio_filename=file.filename or "",
            )
            payload["saved_record"] = {
                "record_id": saved["record_id"],
                "saved_at": saved["saved_at"],
                "message": "Patient analysis saved successfully",
                "patient": saved["patient"],
            }
        else:
            payload["saved_record"] = None

        return payload
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.get("/api/patients/lookup")
def api_lookup_patient(patient_id: str = None, id_number: str = None, q: str = None):
    """Auto-fill helper: find patient by Patient ID or ID card number."""
    query_pid = patient_id or q
    query_nid = id_number or q
    found = find_patient(patient_id=query_pid, id_number=query_nid)
    if not found:
        return {"found": False, "patient": None}
    return {
        "found": True,
        "patient": found["patient"],
        "record_id": found["record_id"],
        "saved_at": found["saved_at"],
        "match_field": found["match_field"],
        "message": "Existing patient found. Details loaded automatically.",
    }


@app.get("/api/patients")
def api_list_patients(patient_id: str = None, limit: int = 50):
    records = list_records(limit=limit, patient_id=patient_id)
    return {"count": len(records), "records": records}


@app.get("/api/patients/{patient_id}/history")
def api_patient_history(patient_id: str):
    history = get_patient_history(patient_id)
    return {"patient_id": patient_id, "count": len(history), "records": history}


@app.get("/api/records/{record_id}")
def api_get_record(record_id: str):
    record = get_record(record_id)
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")
    return record_to_view(record)


@app.delete("/api/records/{record_id}")
def api_delete_record(record_id: str):
    deleted = delete_record(record_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Record not found")
    patient = deleted.get("patient") or {}
    return {
        "ok": True,
        "message": "Record deleted",
        "record_id": record_id,
        "patient_id": patient.get("patient_id") or "",
        "name": patient.get("name") or "",
    }


@app.delete("/api/patients/{patient_id}")
def api_delete_patient(patient_id: str):
    result = delete_patient(patient_id=patient_id)
    if result["deleted_count"] == 0:
        raise HTTPException(status_code=404, detail="No records found for this patient")
    return {
        "ok": True,
        "message": f"Deleted {result['deleted_count']} record(s) for patient {patient_id}",
        **result,
    }


@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND_DIR / "static"), name="static")
