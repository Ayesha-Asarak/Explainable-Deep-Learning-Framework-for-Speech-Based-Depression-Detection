"""Participant-level COVAREP feature extraction from DAIC-WOZ archives."""

from __future__ import annotations

import io
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from .data import _parse_participant_intervals


COVAREP_RATE = 100.0  # frames per second in DAIC-WOZ


def _read_covarep_and_transcript(source: dict):
    if source["kind"] == "file":
        folder = Path(source["path"]).parent
        cov_files = sorted(folder.glob("*_COVAREP.csv"))
        if not cov_files:
            return None, None
        cov = pd.read_csv(cov_files[0], header=None)
        transcript_text = None
        if source.get("transcript"):
            transcript_text = Path(source["transcript"]).read_text(
                encoding="utf-8-sig", errors="replace"
            )
        return cov, transcript_text

    with zipfile.ZipFile(source["path"]) as archive:
        cov_members = [
            name for name in archive.namelist()
            if name.upper().endswith("_COVAREP.CSV")
            and not name.startswith("__MACOSX/")
        ]
        if not cov_members:
            return None, None
        cov = pd.read_csv(io.BytesIO(archive.read(cov_members[0])), header=None)
        transcript_text = None
        if source.get("transcript"):
            transcript_text = archive.read(source["transcript"]).decode(
                "utf-8-sig", errors="replace"
            )
        return cov, transcript_text


def _participant_frame_mask(n_frames: int, intervals) -> np.ndarray:
    mask = np.zeros(n_frames, dtype=bool)
    for start, stop in intervals:
        a = max(0, int(start * COVAREP_RATE))
        b = min(n_frames, int(np.ceil(stop * COVAREP_RATE)))
        if b > a:
            mask[a:b] = True
    return mask


def extract_covarep_vector(source: dict) -> np.ndarray:
    """
    Aggregate COVAREP frames over participant speech into one feature vector.

    Uses voiced frames when VUV is available (column 1 in DAIC-WOZ COVAREP).
    """
    cov, transcript_text = _read_covarep_and_transcript(source)
    if cov is None or cov.shape[0] == 0:
        raise ValueError("missing COVAREP features")

    values = cov.to_numpy(dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    n_frames = values.shape[0]

    if transcript_text:
        intervals = _parse_participant_intervals(transcript_text)
        mask = _participant_frame_mask(n_frames, intervals)
    else:
        mask = np.ones(n_frames, dtype=bool)

    # Column 1 is VUV in DAIC-WOZ COVAREP exports.
    if values.shape[1] > 1:
        voiced = values[:, 1] > 0.5
        use = mask & voiced
        if use.sum() < 50:
            use = mask
    else:
        use = mask

    if use.sum() == 0:
        raise ValueError("no usable COVAREP frames")

    selected = values[use]
    # Drop near-constant columns after selection.
    stats = []
    for reducer in (np.mean, np.std, np.min, np.max, np.median):
        stats.append(reducer(selected, axis=0))
    vector = np.concatenate(stats).astype(np.float32)
    return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
