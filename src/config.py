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
SSL_MODEL_PATH = MODEL_DIR / "depression_ssl.pkl"
SSL_METADATA_PATH = MODEL_DIR / "ssl_training_metadata.json"
ACOUSTIC_MODEL_PATH = MODEL_DIR / "depression_acoustic_candidate.pkl"
ACOUSTIC_METADATA_PATH = MODEL_DIR / "acoustic_candidate_metadata.json"
MULTIMODAL_MODEL_PATH = MODEL_DIR / "depression_multimodal.pkl"
MULTIMODAL_METADATA_PATH = MODEL_DIR / "multimodal_training_metadata.json"
SPLIT_PATH = MODEL_DIR / "participant_split.json"
MANIFEST_PATH = MODEL_DIR / "participant_manifest.json"
EMBEDDING_CACHE_DIR = MODEL_DIR / "embedding_cache"
EMBEDDING_CACHE_DIR.mkdir(exist_ok=True)
ENGINEERED_CACHE_DIR = MODEL_DIR / "engineered_feature_cache"
ENGINEERED_CACHE_DIR.mkdir(exist_ok=True)
ENGINEERED_MODEL_PATH = MODEL_DIR / "depression_engineered_candidate.pkl"
SCALER_PATH = MODEL_DIR / "feature_scaler.pkl"
METADATA_PATH = MODEL_DIR / "training_metadata.json"
PATIENT_RECORDS_DIR = ROOT_DIR / "patient_records"
PATIENT_RECORDS_DIR.mkdir(exist_ok=True)

# Pretrained speech SSL configuration
SSL_MODEL_ID = "microsoft/wavlm-base-plus"
SSL_SEGMENT_DURATION = 5.0
SSL_SEGMENT_OVERLAP = 0.5
SSL_MAX_DURATION = 90.0
SSL_MAX_SEGMENTS = 16
SSL_EMBEDDING_DIM = 768
ACTIVE_MODEL = "acoustic"  # "acoustic", "ssl" or "cnn"

# Official DAIC-WOZ / AVEC 2017 labels and partitions
OFFICIAL_TRAIN_LABELS = ROOT_DIR / "train_split_Depression_AVEC2017.csv"
OFFICIAL_DEV_LABELS = ROOT_DIR / "dev_split_Depression_AVEC2017.csv"
OFFICIAL_TEST_LABELS = ROOT_DIR / "full_test_split.csv"

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
