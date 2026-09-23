# CARE DOLL Voice System

An adaptive, privacy-first personalized voice cloning and recognition system built with SpeechBrain ECAPA-TDNN, Faster-Whisper, and Coqui XTTS-v2.

---

## 1. Quick Start

### Step 1: Create Virtual Environment
```powershell
python -m venv venv
```

### Step 2: Activate Virtual Environment
```powershell
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Requirements
```powershell
pip install -r requirements.txt
```

### Step 4: Run the Application
```powershell
python main.py
```

---

## 2. System Architecture

```text
d:\Voice_cloning\
├── Voice_Input/              # Audio recording and Faster-Whisper speech-to-text
├── Voice_Recognition/        # SpeechBrain ECAPA-TDNN 192-d speaker recognition & enrollment
├── Voice_Cloning/            # XTTS-v2 zero-shot voice cloning pipeline & reference management
├── Voice_Finetuning/         # Adaptive per-user XTTS-v2 fine-tuning subsystem with zero-shot fallback
├── data/                     # Voice profiles, reference audio, generated audio, and transcripts
├── main.py                   # Main system entry point
└── requirements.txt          # Project dependencies
```

### Subsystems Breakdown

### A. Voice Input (`Voice_Input/`)
- Handles audio capture and speech-to-text transcription via **Faster-Whisper** (`WhisperModel`).

### B. Voice Recognition (`Voice_Recognition/`)
- Preprocesses audio to 16 kHz mono with high-pass filtering (70 Hz) and silence trimming.
- Extracts 192-dimensional speaker embeddings using SpeechBrain **ECAPA-TDNN**.
- Manages voice profiles with a duration-aware policy (maintains the top-5 longest valid recordings per user).

### C. Voice Cloning (`Voice_Cloning/`)
- Generates speech using **XTTS-v2 Zero-Shot** cloning.
- Automatically selects the user's top-ranked reference audio for synthesis.

### D. Adaptive Fine-Tuning (`Voice_Finetuning/`)
- **Per-User Personalization**: Trains an isolated candidate checkpoint per user without cross-user contamination.
- **Data Benchmark Trigger**: Starts Phase 1 fine-tuning when a user reaches **50 unique valid recordings** (~30–60 min usable speech).
- **Deterministic 40/10 Split**: Partitions recordings into 40 training and 10 validation samples with zero leakage and locks the validation set across future retraining rounds.
- **Resource Safety**: Designed for 4.0 GB VRAM laptop GPUs (`batch_size=1`, `grad_accum=2`, GPU cache deallocation).
- **Multi-Signal Evaluation**: Compares fine-tuned candidate against the zero-shot baseline on identical validation sentences across:
  1. ECAPA-TDNN speaker cosine similarity
  2. Faster-Whisper Word Error Rate (WER)
  3. Audio signal stability and quality metrics
- **Decision Engine**: Promotes model only if hard quality criteria pass AND speaker similarity improves by $\ge +0.02$. Otherwise, candidate is rejected and zero-shot remains primary.
- **Automatic Fallback**: `PersonalizedVoiceManager` routes to the promoted model; if generation fails at runtime (e.g., CUDA OOM), it automatically falls back to Zero-Shot XTTS.

---

## 3. Running Tests & Pipelines

### Run Full Test Suite (Unit & Subsystem Tests)
```powershell
# Run Voice_Finetuning test suite (26 tests)
python -m unittest discover -s Voice_Finetuning/tests -p "test_*.py" -v

# Run Voice_Recognition test suite
python Voice_Recognition/tests/test_duration_aware_policy.py

# Run Voice_Cloning zero-shot pipeline test
python Voice_Cloning/tests/test_voice_cloning_pipeline.py
```

### Run Adaptive Fine-Tuning Readiness Check
```python
from Voice_Finetuning.pipeline import AdaptiveFineTuningPipeline

pipeline = AdaptiveFineTuningPipeline()
result = pipeline.run_for_user("user_001")

print(f"Readiness: {result.data_status.readiness_status.value}")
print(f"Valid recordings: {result.data_status.valid_recordings_count}/{result.data_status.required_threshold}")
print(f"Active Voice Engine: {result.active_primary_engine}")
```

### Generate Speech with Personalized Voice Manager
```python
from Voice_Finetuning.personalized_voice_manager import PersonalizedVoiceManager

pvm = PersonalizedVoiceManager()
output_path = pvm.generate_for_user(
    user_id="user_001",
    text="Hello, I am your personalized assistant."
)
print(f"Generated audio saved to: {output_path}")
```

---

## 4. Documentation

For full details on the fine-tuning architecture, 50-recording engineering benchmark, evaluation metrics, promotion rules, and rollback procedures, see the comprehensive [Fine-Tuning Guide](file:///d:/Voice_cloning/Voice_Finetuning/docs/FINE_TUNING_GUIDE.md).