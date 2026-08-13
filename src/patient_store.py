"""Store and retrieve patient analysis records."""

from __future__ import annotations

import base64
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .config import ROOT_DIR

RECORDS_DIR = ROOT_DIR / "patient_records"
RECORDS_FILE = RECORDS_DIR / "records.json"
CHARTS_DIR = RECORDS_DIR / "charts"


def _ensure_store() -> None:
    RECORDS_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    if not RECORDS_FILE.exists():
        RECORDS_FILE.write_text("[]", encoding="utf-8")


def _load_all() -> list:
    _ensure_store()
    try:
        data = json.loads(RECORDS_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _save_all(records: list) -> None:
    _ensure_store()
    RECORDS_FILE.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")


def _norm(value) -> str:
    return (value or "").strip()


def _ids_equal(a: str, b: str) -> bool:
    return _norm(a).lower() == _norm(b).lower() and bool(_norm(a))


def _save_charts(record_id: str, charts: dict) -> dict:
    """Save base64 chart images to disk. Returns {chart_name: filename}."""
    _ensure_store()
    saved = {}
    if not charts:
        return saved
    for name, b64 in charts.items():
        if not b64:
            continue
        try:
            raw = base64.b64decode(b64)
        except Exception:
            continue
        filename = f"{record_id}_{name}.png"
        (CHARTS_DIR / filename).write_bytes(raw)
        saved[name] = filename
    return saved


def _load_charts(record: dict) -> dict:
    """Load saved chart PNGs as base64 for the frontend."""
    charts = {}
    analysis = record.get("analysis") or {}
    meta = analysis.get("charts_available") or {}
    record_id = record.get("record_id", "")
    for name, filename in meta.items():
        path = CHARTS_DIR / (filename if filename != "generated" else f"{record_id}_{name}.png")
        if not path.exists():
            path = CHARTS_DIR / f"{record_id}_{name}.png"
        if path.exists():
            charts[name] = base64.b64encode(path.read_bytes()).decode("utf-8")
    return charts


def find_patient(patient_id: str = None, id_number: str = None):
    """
    Find the latest saved patient profile by Patient ID and/or ID card number.
    Returns {patient, record_id, saved_at, match_field} or None.
    """
    pid = _norm(patient_id)
    nid = _norm(id_number)
    if not pid and not nid:
        return None

    for record in _load_all():
        p = record.get("patient") or {}
        existing_pid = _norm(p.get("patient_id"))
        existing_nid = _norm(p.get("id_number"))

        matched_by = None
        if pid and _ids_equal(pid, existing_pid):
            matched_by = "patient_id"
        elif nid and _ids_equal(nid, existing_nid):
            matched_by = "id_number"
        elif pid and _ids_equal(pid, existing_nid):
            matched_by = "id_number"
        elif nid and _ids_equal(nid, existing_pid):
            matched_by = "patient_id"

        if matched_by:
            return {
                "patient": {
                    "name": _norm(p.get("name")),
                    "age": p.get("age"),
                    "id_number": existing_nid,
                    "patient_id": existing_pid,
                    "gender": _norm(p.get("gender")),
                    "phone": _norm(p.get("phone")),
                    "notes": _norm(p.get("notes")),
                },
                "record_id": record.get("record_id"),
                "saved_at": record.get("saved_at"),
                "match_field": matched_by,
            }
    return None


def resolve_patient_identity(patient: dict) -> dict:
    """
    Keep Patient ID and ID number linked for returning patients.
    If either ID matches an existing person, reuse their linked IDs and fill blanks.
    """
    incoming = {
        "name": _norm(patient.get("name")),
        "age": patient.get("age"),
        "id_number": _norm(patient.get("id_number")),
        "patient_id": _norm(patient.get("patient_id")),
        "gender": _norm(patient.get("gender")),
        "phone": _norm(patient.get("phone")),
        "notes": _norm(patient.get("notes")),
    }

    existing = find_patient(
        patient_id=incoming["patient_id"] or None,
        id_number=incoming["id_number"] or None,
    )
    if not existing:
        return incoming

    prev = existing["patient"]

    if incoming["id_number"] and _ids_equal(incoming["id_number"], prev["id_number"]):
        incoming["patient_id"] = prev["patient_id"] or incoming["patient_id"]
        incoming["id_number"] = prev["id_number"]
    elif incoming["patient_id"] and _ids_equal(incoming["patient_id"], prev["patient_id"]):
        incoming["id_number"] = prev["id_number"] or incoming["id_number"]
        incoming["patient_id"] = prev["patient_id"]

    for key in ("name", "gender", "phone", "notes"):
        if not incoming.get(key) and prev.get(key):
            incoming[key] = prev[key]
    if incoming.get("age") in (None, "") and prev.get("age") not in (None, ""):
        incoming["age"] = prev["age"]

    return incoming


def save_patient_record(patient: dict, analysis: dict, audio_filename: str = "") -> dict:
    """Save patient details + analysis result, including explainability charts."""
    patient = resolve_patient_identity(patient)
    record_id = str(uuid.uuid4())[:8]
    now = datetime.now(timezone.utc).isoformat()

    charts_meta = _save_charts(record_id, analysis.get("charts") or {})
    subtype = analysis.get("subtype") or {}

    record = {
        "record_id": record_id,
        "saved_at": now,
        "patient": {
            "name": _norm(patient.get("name")),
            "age": patient.get("age"),
            "id_number": _norm(patient.get("id_number")),
            "patient_id": _norm(patient.get("patient_id")),
            "gender": _norm(patient.get("gender")),
            "phone": _norm(patient.get("phone")),
            "notes": _norm(patient.get("notes")),
        },
        "audio_filename": audio_filename,
        "analysis": {
            "prediction": analysis.get("prediction"),
            "confidence": analysis.get("confidence"),
            "probability_depressed": analysis.get("probability_depressed"),
            "audio_duration_sec": analysis.get("audio_duration_sec"),
            "n_segments": analysis.get("n_segments"),
            "prediction_reason": analysis.get("prediction_reason"),
            "summary": analysis.get("summary"),
            "uncertainty": analysis.get("uncertainty"),
            "key_segment": analysis.get("key_segment"),
            "feature_importance": analysis.get("feature_importance"),
            "subtype": {
                "primary_type": subtype.get("primary_type"),
                "primary_name": subtype.get("primary_name"),
                "primary_description": subtype.get("primary_description", ""),
                "confidence": subtype.get("confidence"),
                "applicable": subtype.get("applicable"),
                "message": subtype.get("message", ""),
                "rankings": subtype.get("rankings", []),
                "matched_symptoms": subtype.get("matched_symptoms", []),
                "disclaimer": subtype.get("disclaimer", ""),
            },
            "acoustic_features": analysis.get("acoustic_features"),
            "segment_explanations": analysis.get("segment_explanations"),
            "timeline_explanations": analysis.get("timeline_explanations"),
            "attribution_method": analysis.get("attribution_method")
            or "segment_occlusion",
            "model_version": analysis.get("model_version") or {},
            "charts_available": charts_meta,
        },
    }

    records = _load_all()
    records.insert(0, record)
    _save_all(records)

    pid = record["patient"]["patient_id"] or record["patient"]["id_number"] or "unknown"
    safe_pid = "".join(c if c.isalnum() or c in "-_" else "_" for c in pid)
    person_file = RECORDS_DIR / f"patient_{safe_pid}.json"
    person_history = []
    if person_file.exists():
        try:
            person_history = json.loads(person_file.read_text(encoding="utf-8"))
            if not isinstance(person_history, list):
                person_history = []
        except (json.JSONDecodeError, OSError):
            person_history = []
    person_history.insert(0, record)
    person_file.write_text(json.dumps(person_history, indent=2, ensure_ascii=False), encoding="utf-8")

    return record


def list_records(limit: int = 50, patient_id: str = None) -> list:
    records = _load_all()
    if patient_id:
        pid = patient_id.strip().lower()
        records = [
            r for r in records
            if (r.get("patient", {}).get("patient_id") or "").lower() == pid
            or (r.get("patient", {}).get("id_number") or "").lower() == pid
        ]
    return records[:limit]


def get_record(record_id: str):
    for r in _load_all():
        if r.get("record_id") == record_id:
            return r
    return None


def record_to_view(record: dict) -> dict:
    """Convert a saved record into the same shape used by the live analysis UI."""
    analysis = record.get("analysis") or {}
    charts = _load_charts(record)
    subtype = analysis.get("subtype") or {}

    return {
        "prediction": analysis.get("prediction"),
        "confidence": analysis.get("confidence") or 0,
        "probability_depressed": analysis.get("probability_depressed") or 0,
        "audio_duration_sec": analysis.get("audio_duration_sec") or 0,
        "n_segments": analysis.get("n_segments") or 0,
        "prediction_reason": analysis.get("prediction_reason") or "",
        "summary": analysis.get("summary") or "",
        "uncertainty": analysis.get("uncertainty"),
        "key_segment": analysis.get("key_segment"),
        "feature_importance": analysis.get("feature_importance") or [],
        "timeline_explanations": analysis.get("timeline_explanations") or [],
        "segment_explanations": analysis.get("segment_explanations") or [],
        "acoustic_features": analysis.get("acoustic_features") or {},
        "subtype": {
            "primary_type": subtype.get("primary_type"),
            "primary_name": subtype.get("primary_name") or "Not Applicable",
            "primary_description": subtype.get("primary_description", ""),
            "confidence": subtype.get("confidence") or 0,
            "applicable": subtype.get("applicable", False),
            "message": subtype.get("message") or "",
            "rankings": subtype.get("rankings") or [],
            "matched_symptoms": subtype.get("matched_symptoms") or [],
            "disclaimer": subtype.get("disclaimer") or "",
        },
        "patient": record.get("patient") or {},
        "saved_record": {
            "record_id": record.get("record_id"),
            "saved_at": record.get("saved_at"),
            "message": "Loaded from saved patient records",
        },
        "charts": {
            "spectrogram": charts.get("spectrogram"),
            "grad_cam": charts.get("grad_cam"),
            "timeline": charts.get("timeline"),
            "features": charts.get("features"),
            "subtype": charts.get("subtype"),
        },
        "has_charts": bool(charts),
        "from_saved_record": True,
        "attribution_method": analysis.get("attribution_method")
        or "segment_occlusion",
        "model_version": analysis.get("model_version") or {"active_model": "acoustic"},
    }


def get_patient_history(patient_id: str) -> list:
    return list_records(limit=100, patient_id=patient_id)


def _delete_chart_files(record: dict) -> int:
    """Remove chart PNG files for one record. Returns count deleted."""
    deleted = 0
    analysis = record.get("analysis") or {}
    meta = analysis.get("charts_available") or {}
    record_id = record.get("record_id", "")
    filenames = set()
    for name, filename in meta.items():
        if filename and filename != "generated":
            filenames.add(filename)
        filenames.add(f"{record_id}_{name}.png")
    for filename in filenames:
        path = CHARTS_DIR / filename
        if path.exists():
            try:
                path.unlink()
                deleted += 1
            except OSError:
                pass
    return deleted


def _sync_person_file(patient: dict, remaining_for_patient: list) -> None:
    """Rewrite or remove the per-patient history JSON after deletes."""
    pid = _norm((patient or {}).get("patient_id")) or _norm((patient or {}).get("id_number"))
    if not pid:
        return
    safe_pid = "".join(c if c.isalnum() or c in "-_" else "_" for c in pid)
    person_file = RECORDS_DIR / f"patient_{safe_pid}.json"
    if not remaining_for_patient:
        if person_file.exists():
            try:
                person_file.unlink()
            except OSError:
                pass
        return
    person_file.write_text(
        json.dumps(remaining_for_patient, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def delete_record(record_id: str) -> dict | None:
    """Delete one analysis record and its charts. Returns deleted record or None."""
    records = _load_all()
    target = None
    kept = []
    for record in records:
        if record.get("record_id") == record_id:
            target = record
        else:
            kept.append(record)
    if target is None:
        return None

    _delete_chart_files(target)
    _save_all(kept)

    patient = target.get("patient") or {}
    pid = _norm(patient.get("patient_id")) or _norm(patient.get("id_number"))
    remaining = [
        r for r in kept
        if _ids_equal(_norm((r.get("patient") or {}).get("patient_id")), pid)
        or _ids_equal(_norm((r.get("patient") or {}).get("id_number")), pid)
    ] if pid else []
    _sync_person_file(patient, remaining)
    return target


def delete_patient(patient_id: str = None, id_number: str = None) -> dict:
    """
    Delete all saved records for a patient (by Patient ID and/or ID number).
    Returns {deleted_count, record_ids}.
    """
    pid = _norm(patient_id)
    nid = _norm(id_number)
    if not pid and not nid:
        return {"deleted_count": 0, "record_ids": []}

    records = _load_all()
    kept = []
    deleted = []
    for record in records:
        p = record.get("patient") or {}
        existing_pid = _norm(p.get("patient_id"))
        existing_nid = _norm(p.get("id_number"))
        match = (
            (pid and (_ids_equal(pid, existing_pid) or _ids_equal(pid, existing_nid)))
            or (nid and (_ids_equal(nid, existing_nid) or _ids_equal(nid, existing_pid)))
        )
        if match:
            deleted.append(record)
            _delete_chart_files(record)
        else:
            kept.append(record)

    _save_all(kept)

    # Remove matching per-patient history files
    for record in deleted:
        _sync_person_file(record.get("patient") or {}, [])

    return {
        "deleted_count": len(deleted),
        "record_ids": [r.get("record_id") for r in deleted],
    }
