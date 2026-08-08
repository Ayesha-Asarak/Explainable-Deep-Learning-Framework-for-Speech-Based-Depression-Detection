"""Dataset loading for DAIC-WOZ style interview recordings."""

import csv
import hashlib
import io
from pathlib import Path
import json
import re
import zipfile

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import (
    DATA_DIR,
    DEPRESSED_DIR,
    NON_DEPRESSED_DIR,
    OFFICIAL_DEV_LABELS,
    OFFICIAL_TEST_LABELS,
    OFFICIAL_TRAIN_LABELS,
    SAMPLE_RATE,
    SEGMENT_DURATION,
    SEGMENT_OVERLAP,
)
from .features import (
    load_audio,
    preprocess_audio,
    segment_audio,
    compute_mel_spectrogram,
    extract_acoustic_features,
    features_to_vector,
    augment_audio,
)


def discover_audio_files(data_dir: Path) -> list[tuple[Path, int, str]]:
    """Return extracted (audio_path, label, participant_id) tuples."""
    samples = []
    for label, folder in [(1, DEPRESSED_DIR), (0, NON_DEPRESSED_DIR)]:
        folder_path = data_dir / folder
        if not folder_path.exists():
            continue
        for participant_dir in sorted(folder_path.iterdir()):
            if not participant_dir.is_dir():
                continue
            audio_files = list(participant_dir.glob("*_AUDIO.wav"))
            if audio_files:
                pid = participant_dir.name.replace("_P", "")
                samples.append((audio_files[0], label, pid))
    return samples


def _participant_id(path: Path) -> str:
    """Normalize names such as '381_P 2' and '381_P.zip' to '381'."""
    match = re.match(r"(\d+)", path.stem)
    return match.group(1) if match else path.stem.replace("_P", "").strip()


def load_official_label_metadata() -> dict[str, dict]:
    """Load official PHQ binary labels and AVEC partitions when available."""
    metadata = {}
    files = (
        ("train", OFFICIAL_TRAIN_LABELS),
        ("dev", OFFICIAL_DEV_LABELS),
        ("test", OFFICIAL_TEST_LABELS),
    )
    for split_name, path in files:
        if not path.exists():
            continue
        with open(path, newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                participant = row.get("Participant_ID") or row.get(
                    "participant_ID"
                )
                binary = row.get("PHQ8_Binary") or row.get("PHQ_Binary")
                score = row.get("PHQ8_Score") or row.get("PHQ_Score")
                gender_value = row.get("Gender")
                if participant is None or binary in (None, ""):
                    continue
                try:
                    pid = str(int(float(participant)))
                    label = int(float(binary))
                    phq_score = (
                        float(score) if score not in (None, "") else None
                    )
                    gender = (
                        int(float(gender_value))
                        if gender_value not in (None, "")
                        else None
                    )
                except (TypeError, ValueError):
                    continue
                metadata[pid] = {
                    "label": label,
                    "phq_score": phq_score,
                    "gender": gender,
                    "official_split": split_name,
                    "label_source": str(path),
                }
                if label not in (0, 1):
                    raise ValueError(
                        f"Invalid official binary PHQ label for participant "
                        f"{pid}: {label}"
                    )
                if phq_score is not None and not 0 <= phq_score <= 24:
                    raise ValueError(
                        f"Invalid official PHQ score for participant {pid}: "
                        f"{phq_score}"
                    )
    return metadata


def discover_audio_sources(data_dir: Path, include_zips: bool = True) -> tuple[list[dict], list[str]]:
    """
    Discover extracted WAVs and WAVs stored inside participant ZIP archives.

    Extracted WAVs are preferred when both formats exist. Participant IDs found
    under both class folders are excluded to prevent contradictory labels.
    Returns (sources, conflicting_participant_ids).
    """
    by_label = {1: {}, 0: {}}

    for label, folder in [(1, DEPRESSED_DIR), (0, NON_DEPRESSED_DIR)]:
        folder_path = data_dir / folder
        if not folder_path.exists():
            continue

        # Register ZIP sources first; extracted WAVs below replace duplicates.
        if include_zips:
            for archive_path in sorted(folder_path.glob("*.zip")):
                pid = _participant_id(archive_path)
                try:
                    with zipfile.ZipFile(archive_path) as archive:
                        members = [
                            name for name in archive.namelist()
                            if name.upper().endswith("_AUDIO.WAV")
                            and not name.startswith("__MACOSX/")
                        ]
                        transcripts = [
                            name for name in archive.namelist()
                            if name.upper().endswith("_TRANSCRIPT.CSV")
                            and not name.startswith("__MACOSX/")
                        ]
                except (OSError, zipfile.BadZipFile) as exc:
                    print(f"Warning: skipping unreadable ZIP {archive_path}: {exc}")
                    continue

                if members:
                    by_label[label][pid] = {
                        "kind": "zip",
                        "path": archive_path,
                        "member": members[0],
                        "transcript": transcripts[0] if transcripts else None,
                        "label": label,
                        "participant_id": pid,
                    }

        for participant_dir in sorted(folder_path.iterdir()):
            if not participant_dir.is_dir():
                continue
            audio_files = sorted(participant_dir.glob("*_AUDIO.wav"))
            if audio_files:
                pid = _participant_id(participant_dir)
                if (
                    pid in by_label[label]
                    and by_label[label][pid]["kind"] == "file"
                ):
                    print(
                        f"Warning: ignoring duplicate extracted folder "
                        f"{participant_dir.name} for participant {pid}"
                    )
                    continue
                transcript_files = sorted(participant_dir.glob("*_TRANSCRIPT.csv"))
                by_label[label][pid] = {
                    "kind": "file",
                    "path": audio_files[0],
                    "member": None,
                    "transcript": transcript_files[0] if transcript_files else None,
                    "label": label,
                    "participant_id": pid,
                }

    conflicts = sorted(set(by_label[1]) & set(by_label[0]))
    if conflicts:
        print(
            "Warning: excluding participant IDs present in both classes: "
            + ", ".join(conflicts)
        )
        for pid in conflicts:
            by_label[1].pop(pid, None)
            by_label[0].pop(pid, None)

    sources = []
    for label in (1, 0):
        sources.extend(by_label[label][pid] for pid in sorted(by_label[label]))

    # Folder names are only storage locations. Official PHQ labels are the
    # sole ground truth and always override folder-derived labels.
    official = load_official_label_metadata()
    if not official:
        raise FileNotFoundError(
            "Official PHQ label CSV files are required. Expected "
            "train_split_Depression_AVEC2017.csv, "
            "dev_split_Depression_AVEC2017.csv and full_test_split.csv."
        )

    mismatch_count = 0
    officially_labelled = []
    missing_official = []
    for source in sources:
        item = official.get(source["participant_id"])
        source["folder_label"] = int(source["label"])
        if item is None:
            missing_official.append(source["participant_id"])
            continue
        if int(source["label"]) != int(item["label"]):
            mismatch_count += 1
        source.update(item)
        officially_labelled.append(source)

    if missing_official:
        print(
            "Warning: excluding audio without an official PHQ label: "
            + ", ".join(sorted(missing_official))
        )
    sources = officially_labelled
    if mismatch_count:
        print(
            f"Official PHQ labels override {mismatch_count} mismatched "
            "folder labels."
        )
    return sources, conflicts


def _parse_participant_intervals(text: str) -> list[tuple[float, float]]:
    """Read DAIC-WOZ TSV transcript and return Participant speaking intervals."""
    intervals = []
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    for row in reader:
        if (row.get("speaker") or "").strip().lower() != "participant":
            continue
        try:
            start = float(row["start_time"])
            stop = float(row["stop_time"])
        except (KeyError, TypeError, ValueError):
            continue
        if stop > start:
            intervals.append((start, stop))
    return intervals


def parse_participant_transcript(text: str) -> str:
    """Return only participant utterances from a DAIC-WOZ TSV transcript."""
    utterances = []
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    for row in reader:
        if (row.get("speaker") or "").strip().lower() != "participant":
            continue
        value = (row.get("value") or "").strip()
        if value:
            utterances.append(value)
    return " ".join(utterances)


def load_participant_transcript(source: dict) -> str:
    """Load participant-only transcript text from an extracted file or ZIP."""
    transcript = source.get("transcript")
    if not transcript:
        return ""
    try:
        if source["kind"] == "file":
            text = Path(transcript).read_text(
                encoding="utf-8-sig", errors="replace"
            )
        else:
            with zipfile.ZipFile(source["path"]) as archive:
                text = archive.read(transcript).decode(
                    "utf-8-sig", errors="replace"
                )
    except (OSError, KeyError, zipfile.BadZipFile):
        return ""
    return parse_participant_transcript(text)


def _participant_only_audio(
    y: np.ndarray,
    intervals: list[tuple[float, float]],
    max_duration=None,
) -> np.ndarray:
    """Concatenate only participant turns, capped by participant speech duration."""
    chunks = []
    remaining_samples = (
        int(max_duration * SAMPLE_RATE) if max_duration is not None else None
    )

    for start, stop in intervals:
        start_sample = max(0, int(start * SAMPLE_RATE))
        stop_sample = min(len(y), int(stop * SAMPLE_RATE))
        if stop_sample <= start_sample:
            continue
        chunk = y[start_sample:stop_sample]
        if remaining_samples is not None:
            if remaining_samples <= 0:
                break
            chunk = chunk[:remaining_samples]
            remaining_samples -= len(chunk)
        if len(chunk):
            chunks.append(chunk)

    if not chunks:
        return np.array([], dtype=np.float32)
    return preprocess_audio(np.concatenate(chunks), trim=True, normalize=True)


def find_transcript_for_participant(participant_id):
    """Locate a DAIC-WOZ transcript for a participant id under local data folders."""
    if participant_id is None:
        return None
    pid = str(participant_id).strip()
    if not pid:
        return None
    # Accept values like "325", "325_P", "325_AUDIO.wav"
    digits = "".join(ch for ch in pid if ch.isdigit())
    if not digits:
        return None
    candidates = [
        DATA_DIR / DEPRESSED_DIR / f"{digits}_P" / f"{digits}_TRANSCRIPT.csv",
        DATA_DIR / NON_DEPRESSED_DIR / f"{digits}_P" / f"{digits}_TRANSCRIPT.csv",
        DATA_DIR / DEPRESSED_DIR / f"{digits}_TRANSCRIPT.csv",
        DATA_DIR / NON_DEPRESSED_DIR / f"{digits}_TRANSCRIPT.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def apply_participant_only_from_transcript(y, transcript_path, max_duration=None):
    """Filter a full interview waveform to participant turns using a transcript."""
    text = Path(transcript_path).read_text(encoding="utf-8-sig", errors="replace")
    intervals = _parse_participant_intervals(text)
    if not intervals:
        return preprocess_audio(y, trim=True, normalize=True)
    participant_audio = _participant_only_audio(
        y, intervals, max_duration=max_duration
    )
    if len(participant_audio):
        return participant_audio
    if max_duration is not None:
        y = y[: int(max_duration * SAMPLE_RATE)]
    return preprocess_audio(y, trim=True, normalize=True)


def load_audio_source(
    source: dict,
    max_duration=None,
    participant_only: bool = True,
) -> np.ndarray:
    """
    Load an extracted WAV or read one WAV directly from a ZIP into memory.

    ZIP contents are never permanently extracted, keeping disk usage constant.
    """
    transcript_text = None
    if source["kind"] == "file":
        if participant_only and source.get("transcript"):
            transcript_text = Path(source["transcript"]).read_text(
                encoding="utf-8-sig", errors="replace"
            )
        if transcript_text:
            y = load_audio(
                str(source["path"]),
                max_duration=None,
                trim=False,
                normalize=False,
            )
        else:
            return load_audio(str(source["path"]), max_duration=max_duration)
    else:
        with zipfile.ZipFile(source["path"]) as archive:
            audio_bytes = archive.read(source["member"])
            if participant_only and source.get("transcript"):
                transcript_text = archive.read(source["transcript"]).decode(
                    "utf-8-sig", errors="replace"
                )
        y = load_audio(
            io.BytesIO(audio_bytes),
            max_duration=None if transcript_text else max_duration,
            trim=not bool(transcript_text),
            normalize=not bool(transcript_text),
        )

    if transcript_text:
        participant_audio = _participant_only_audio(
            y,
            _parse_participant_intervals(transcript_text),
            max_duration=max_duration,
        )
        if len(participant_audio):
            return participant_audio

    # Fallback for a missing/invalid transcript.
    if max_duration is not None:
        y = y[:int(max_duration * SAMPLE_RATE)]
    return preprocess_audio(y, trim=True, normalize=True)


def describe_audio_source(source: dict) -> str:
    if source["kind"] == "zip":
        return f"{source['path']}::{source['member']}"
    return str(source["path"])


def _source_fingerprint(source: dict) -> str:
    """Stable content fingerprint for cache keys (path + size + mtime)."""
    path = Path(source["path"])
    try:
        stat = path.stat()
        size = stat.st_size
        mtime = int(stat.st_mtime)
    except OSError:
        size, mtime = 0, 0
    member = source.get("member") or ""
    transcript = str(source.get("transcript") or "")
    raw = f"{source['kind']}|{path}|{member}|{transcript}|{size}|{mtime}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _transcript_coverage(source: dict) -> dict:
    """Summarize participant-turn coverage from the transcript."""
    text = None
    try:
        if source["kind"] == "file" and source.get("transcript"):
            text = Path(source["transcript"]).read_text(
                encoding="utf-8-sig", errors="replace"
            )
        elif source["kind"] == "zip" and source.get("transcript"):
            with zipfile.ZipFile(source["path"]) as archive:
                text = archive.read(source["transcript"]).decode(
                    "utf-8-sig", errors="replace"
                )
    except (OSError, zipfile.BadZipFile, KeyError):
        text = None

    if not text:
        return {
            "has_transcript": False,
            "n_participant_turns": 0,
            "participant_speech_sec": 0.0,
        }

    intervals = _parse_participant_intervals(text)
    speech_sec = float(sum(max(0.0, stop - start) for start, stop in intervals))
    return {
        "has_transcript": True,
        "n_participant_turns": len(intervals),
        "participant_speech_sec": round(speech_sec, 2),
    }


def build_participant_manifest(
    data_dir: Path,
    include_zips: bool = True,
) -> tuple[list[dict], list[str]]:
    """
    Canonical participant list with provenance for leakage-safe experiments.
    """
    sources, conflicts = discover_audio_sources(data_dir, include_zips=include_zips)
    manifest = []
    for source in sources:
        coverage = _transcript_coverage(source)
        path = Path(source["path"])
        try:
            size_bytes = path.stat().st_size
        except OSError:
            size_bytes = 0
        entry = {
            "participant_id": source["participant_id"],
            "label": int(source["label"]),
            "folder_label": int(
                source.get("folder_label", source["label"])
            ),
            "official_split": source.get("official_split"),
            "phq_score": source.get("phq_score"),
            "gender": source.get("gender"),
            "label_source": source.get("label_source", "folder"),
            "kind": source["kind"],
            "path": str(source["path"]),
            "member": source.get("member"),
            "transcript": str(source["transcript"]) if source.get("transcript") else None,
            "source_fingerprint": _source_fingerprint(source),
            "size_bytes": size_bytes,
            **coverage,
        }
        manifest.append(entry)
    return manifest, conflicts


def fixed_participant_split(
    manifest: list[dict],
    test_ratio: float = 0.22,
    seed: int = 42,
) -> dict:
    """
    Stratified participant split. The held-out test participants must never be
    used for model/threshold selection.
    """
    by_label = {0: [], 1: []}
    for entry in manifest:
        by_label[int(entry["label"])].append(entry["participant_id"])

    rng = np.random.RandomState(seed)
    test_pids = []
    train_pids = []
    for label in (0, 1):
        pids = sorted(by_label[label])
        n_test = max(1, int(round(len(pids) * test_ratio)))
        chosen = rng.choice(pids, size=n_test, replace=False).tolist()
        test_pids.extend(chosen)
        train_pids.extend([pid for pid in pids if pid not in chosen])

    test_pids = sorted(test_pids)
    train_pids = sorted(train_pids)
    label_lookup = {e["participant_id"]: int(e["label"]) for e in manifest}
    return {
        "seed": seed,
        "test_ratio": test_ratio,
        "train_participant_ids": train_pids,
        "test_participant_ids": test_pids,
        "train_labels": [label_lookup[pid] for pid in train_pids],
        "test_labels": [label_lookup[pid] for pid in test_pids],
        "n_train": len(train_pids),
        "n_test": len(test_pids),
    }


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


def load_json(path: Path):
    with open(path) as handle:
        return json.load(handle)


def build_segment_dataset(
    data_dir: Path,
    augment: bool = False,
    augment_factor: int = 3,
    max_duration: float = 180.0,
    max_segments_per_file=None,
    include_zips: bool = True,
    participant_only: bool = True,
) -> tuple[list[np.ndarray], list[np.ndarray], list[dict]]:
    """
    Build mel spectrogram segments and acoustic feature vectors.
    Returns (spectrograms, labels, metadata_per_segment).
    """
    spectrograms, labels, metadata = [], [], []
    sources, conflicts = discover_audio_sources(data_dir, include_zips=include_zips)
    print(
        f"Discovered {len(sources)} unique participants "
        f"({sum(s['kind'] == 'zip' for s in sources)} from ZIP, "
        f"{sum(s['kind'] == 'file' for s in sources)} extracted)"
    )

    for source_number, source in enumerate(sources, start=1):
        label = source["label"]
        pid = source["participant_id"]
        print(
            f"  [{source_number}/{len(sources)}] "
            f"{'depressed' if label else 'non-depressed'} participant {pid} "
            f"({source['kind']})"
        )
        try:
            y = load_audio_source(
                source,
                max_duration=max_duration,
                participant_only=participant_only,
            )
        except Exception as exc:
            print(f"Warning: skipping participant {pid}: {exc}")
            continue

        segments = segment_audio(y, SEGMENT_DURATION, SEGMENT_OVERLAP)
        if max_segments_per_file and len(segments) > max_segments_per_file:
            indices = np.linspace(0, len(segments) - 1, max_segments_per_file, dtype=int)
            segments = [segments[i] for i in indices]

        variants = [segments]
        if augment:
            for _ in range(augment_factor):
                aug_segments = []
                for seg in segments:
                    aug_y = augment_audio(seg)
                    aug_segments.append(aug_y[: len(seg)] if len(aug_y) >= len(seg) else np.pad(aug_y, (0, len(seg) - len(aug_y))))
                variants.append(aug_segments)

        for variant_idx, segs in enumerate(variants):
            for seg_idx, seg in enumerate(segs):
                spec = compute_mel_spectrogram(seg)
                feats = extract_acoustic_features(seg)
                spectrograms.append(spec)
                labels.append(label)
                metadata.append({
                    "participant_id": pid,
                    "audio_source": describe_audio_source(source),
                    "source_kind": source["kind"],
                    "participant_only": participant_only and bool(source.get("transcript")),
                    "segment_idx": seg_idx,
                    "variant": variant_idx,
                    "label": label,
                    "acoustic_features": features_to_vector(feats),
                })
    return spectrograms, labels, metadata


class MelSpectrogramDataset(Dataset):
    def __init__(self, spectrograms: list[np.ndarray], labels: list[int]):
        self.spectrograms = spectrograms
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.spectrograms[idx])
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


def participant_level_split(
    metadata: list[dict], test_ratio: float = 0.22
) -> tuple[list[int], list[int]]:
    """Stratified participant split to avoid speaker leakage."""
    participants = {}
    for i, m in enumerate(metadata):
        pid = m["participant_id"]
        participants.setdefault(pid, {"indices": [], "label": m["label"]})
        participants[pid]["indices"].append(i)

    rng = np.random.RandomState(42)
    test_pids = set()
    for label in (0, 1):
        pids = sorted(
            pid for pid, info in participants.items() if info["label"] == label
        )
        n_test = max(1, int(round(len(pids) * test_ratio)))
        test_pids.update(rng.choice(pids, size=n_test, replace=False).tolist())

    train_idx, test_idx = [], []
    for pid, info in participants.items():
        if pid in test_pids:
            test_idx.extend(info["indices"])
        else:
            train_idx.extend(info["indices"])
    return train_idx, test_idx


def save_training_metadata(path: Path, info: dict) -> None:
    with open(path, "w") as f:
        json.dump(info, f, indent=2)
