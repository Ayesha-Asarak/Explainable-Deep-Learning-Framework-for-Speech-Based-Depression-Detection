from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR
MODEL_DIR = ROOT_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

SAMPLE_RATE = 16000
N_MELS = 128
N_MFCC = 13
HOP_LENGTH = 512
N_FFT = 2048
SEGMENT_DURATION = 3.0  # seconds
SEGMENT_OVERLAP = 0.5   # 50% overlap

DEPRESSED_DIR = "depressed"
NON_DEPRESSED_DIR = "non depressed"

MODEL_PATH = MODEL_DIR / "depression_cnn.pt"
SCALER_PATH = MODEL_DIR / "feature_scaler.pkl"
METADATA_PATH = MODEL_DIR / "training_metadata.json"

FEATURE_NAMES = [
    "pitch_mean_hz",
    "pitch_std_hz",
    "energy_mean",
    "energy_std",
    "speech_rate",
    "pause_ratio",
    "zero_crossing_rate",
    "spectral_centroid",
    "spectral_rolloff",
    "spectral_bandwidth",
    "mfcc_1_mean",
    "mfcc_2_mean",
    "mfcc_3_mean",
    "mfcc_4_mean",
    "mfcc_5_mean",
    "mfcc_6_mean",
    "mfcc_7_mean",
    "mfcc_8_mean",
    "mfcc_9_mean",
    "mfcc_10_mean",
    "mfcc_11_mean",
    "mfcc_12_mean",
    "mfcc_13_mean",
]
