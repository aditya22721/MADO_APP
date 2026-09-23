"""
Configuration settings for Voice_Finetuning Subsystem.

Defines project paths, engineering thresholds, audio validation bounds,
deterministic train/val split ratios, multi-signal evaluation parameters,
and resource-safe XTTS-v2 fine-tuning hyperparameters.
"""

from pathlib import Path
import torch

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FINETUNING_DIRECTORY = PROJECT_ROOT / "Voice_Finetuning"
DATASETS_DIRECTORY = FINETUNING_DIRECTORY / "datasets"
CHECKPOINTS_DIRECTORY = FINETUNING_DIRECTORY / "checkpoints"
RESULTS_DIRECTORY = FINETUNING_DIRECTORY / "results"
LOGS_DIRECTORY = FINETUNING_DIRECTORY / "logs"

# References to existing system data paths
VOICE_PROFILES_DIRECTORY = PROJECT_ROOT / "data" / "voice_profiles"
TRANSCRIPTS_DIRECTORY = PROJECT_ROOT / "data" / "text"
VOICE_CLONING_DATA_DIRECTORY = PROJECT_ROOT / "data" / "voice_cloning"
GENERATED_AUDIO_DIRECTORY = VOICE_CLONING_DATA_DIRECTORY / "generated_audio"

# ============================================================
# ENGINEERING DATA THRESHOLDS
# ============================================================

# Note: 50 recordings is the project engineering benchmark,
# not an official XTTS-v2 minimum requirement.
PHASE_1_RECORDING_THRESHOLD = 50
BATCH_INCREMENT = 25
MAX_TRAINING_ATTEMPTS = 5

TARGET_SPEECH_DURATION_MINUTES_MIN = 30.0
TARGET_SPEECH_DURATION_MINUTES_MAX = 60.0

# ============================================================
# RECORDING VALIDATION BOUNDS
# ============================================================

MIN_RECORDING_DURATION_SEC = 2.0
MAX_RECORDING_DURATION_SEC = 15.0
MIN_USABLE_SPEECH_SEC = 2.0
MAX_SILENCE_RATIO = 0.60
MAX_CLIPPING_RATIO = 0.01

XTTS_SAMPLE_RATE = 22050
XTTS_OUTPUT_SAMPLE_RATE = 24000
ECAPA_SAMPLE_RATE = 16000

SUPPORTED_AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".m4a"
}

# ============================================================
# DETERMINISTIC TRAIN / EVAL SPLIT
# ============================================================

PHASE_1_TRAIN_COUNT = 40
PHASE_1_VAL_COUNT = 10
DEFAULT_VAL_SPLIT_RATIO = 0.20
SPLIT_RANDOM_SEED = 42

# ============================================================
# EVALUATION & ACCEPTANCE THRESHOLDS
# ============================================================

MIN_SPEAKER_SIMILARITY = 0.72
MAX_WER = 0.20
MIN_IMPROVEMENT_MARGIN = 0.02
MIN_QUALITY_SCORE = 0.70

# ============================================================
# XTTS MODEL & DEVICE SETTINGS
# ============================================================

XTTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
DEFAULT_LANGUAGE = "en"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Safety settings for 4GB VRAM GPU
DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 1
DEFAULT_GRAD_ACCUM_STEPS = 2
DEFAULT_LEARNING_RATE = 5e-6
MAX_AUDIO_LENGTH = 255995  # ~11.6 seconds @ 22050 Hz

# Base model download links
DVAE_CHECKPOINT_LINK = "https://huggingface.co/coqui/XTTS-v2/resolve/main/dvae.pth"
MEL_NORM_LINK = "https://huggingface.co/coqui/XTTS-v2/resolve/main/mel_stats.pth"
TOKENIZER_FILE_LINK = "https://huggingface.co/coqui/XTTS-v2/resolve/main/vocab.json"
XTTS_CHECKPOINT_LINK = "https://huggingface.co/coqui/XTTS-v2/resolve/main/model.pth"
XTTS_CONFIG_LINK = "https://huggingface.co/coqui/XTTS-v2/resolve/main/config.json"

# ============================================================
# DIRECTORY CREATION
# ============================================================

REQUIRED_DIRECTORIES = [
    DATASETS_DIRECTORY,
    CHECKPOINTS_DIRECTORY,
    RESULTS_DIRECTORY,
    LOGS_DIRECTORY,
]


def create_required_directories():
    """Create all required directories for the fine-tuning subsystem."""
    for directory in REQUIRED_DIRECTORIES:
        directory.mkdir(parents=True, exist_ok=True)
