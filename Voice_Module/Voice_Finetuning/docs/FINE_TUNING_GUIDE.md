# Adaptive Per-User XTTS-v2 Fine-Tuning & Personalization Guide

## 1. Why Fine-Tuning is Needed
While zero-shot XTTS-v2 produces recognizable voice clones from a single short reference audio, zero-shot synthesis can exhibit variability in acoustic timbre, subtle accent reproduction, expressive prosody, and phonetic nuances.

Personalized per-user fine-tuning adapts the internal autoregressive GPT decoder of XTTS-v2 specifically to a user's vocal characteristics, speech cadences, and pronunciation patterns. This significantly increases speaker similarity and consistency over long-form conversations.

---

## 2. Why Zero-Shot XTTS Remains Available
Zero-shot XTTS is the indispensable baseline and universal safety fallback:
1. **Cold Start**: Enables immediate voice cloning from the moment an enrollment recording is provided, before sufficient recordings exist for fine-tuning.
2. **Quality Baseline**: Provides an objective benchmark against which candidate fine-tuned models are evaluated on identical validation sentences.
3. **Runtime Fallback**: If a fine-tuned model encounters an unexpected runtime error or OOM condition during synthesis, `PersonalizedVoiceManager` automatically falls back to Zero-Shot XTTS, guaranteeing uninterrupted voice response.

---

## 3. Data Engineering Benchmark: The 50-Recording Trigger
> [!IMPORTANT]
> **Project Engineering Benchmark vs. Official XTTS Requirement**:
> `50 valid recordings` (~30–60 minutes of usable speech) is established as the **PROJECT ENGINEERING BENCHMARK** for triggering Phase 1 fine-tuning in this CARE DOLL system. It is **NOT** an official universal minimum defined by the authors of XTTS-v2.

Why 50 recordings?
- Prevents overfitting or catastrophic forgetting on tiny, unrepresentative datasets.
- Ensures sufficient acoustic and phonetic variety (vowels, consonants, pitch variations).
- If a user has fewer than 50 recordings, the system strictly reports `NOT_READY` and keeps zero-shot active. **No fake or duplicated data is ever generated to artificially satisfy the threshold.**

---

## 4. What Constitutes a Valid Recording?
Before any audio file is admitted into a user's fine-tuning dataset, `DataManager` verifies:
- **Format & Readability**: Supported audio container (`.wav`, `.mp3`, `.flac`) that decodes without corruption.
- **Channels & Sample Rate**: Mono compatibility (stereo is downmixed to mono) and convertible sample rate.
- **Duration Limits**: Raw audio duration between `2.0` and `15.0` seconds.
- **Usable Speech**: At least `2.0` seconds of active speech detected by `SpeakerAudioPreprocessor`.
- **Signal Quality**:
  - Max clipping ratio $\le 1.0\%$
  - Max silence ratio $\le 60.0\%$
  - No `NaN` or `Inf` floating-point values.
- **Cryptographic Deduplication**: SHA-256 hash checks prevent identical or renamed duplicate files from inflating recording counts.

---

## 5. Deterministic Train / Validation Split
To prevent data leakage:
- **Phase 1 Split**: 50 valid recordings $\rightarrow$ **40 training** and **10 validation**.
- **Deterministic**: Controlled via fixed seed (`SPLIT_RANDOM_SEED = 42`).
- **Zero Overlap**: Training and validation sets share zero audio files.
- **Validation Stability**: Across retraining rounds (Round 1 $\rightarrow$ Round 2 $\rightarrow$ Round 3), the initial 10 validation recordings remain locked in the validation set. All additional recordings (batches of 25–50) expand the training set.

---

## 6. Multi-Signal Objective Evaluation
Evaluation does not rely on a single subjective number; it combines three independent objective signals evaluated on identical validation sentences:

1. **Speaker Similarity**:
   $$\text{Similarity} = \frac{\mathbf{e}_{\text{gen}} \cdot \mathbf{e}_{\text{ref}}}{\|\mathbf{e}_{\text{gen}}\| \|\mathbf{e}_{\text{ref}}\|}$$
   Computed using SpeechBrain's ECAPA-TDNN 192-dimensional embeddings against the user's reference embeddings.
2. **Content Intelligibility (WER / CER)**:
   Faster-Whisper transcribes synthesized audio and computes Word Error Rate against expected validation ground truth.
3. **Audio Stability & Quality**:
   Checks clipping ratio, silence ratio, audio duration fidelity, and absence of numerical anomalies.

---

## 7. Model Promotion Criteria
A candidate model is promoted if and only if **BOTH** conditions are met:
1. **Hard Minimum Quality Thresholds**:
   - Mean Speaker Similarity $\ge 0.72$
   - Mean Word Error Rate (WER) $\le 0.20$
   - Audio Quality Score $\ge 0.70$
   - Validation Clipping Ratio $\le 2.0\%$
2. **Meaningful Improvement Margin**:
   - $\Delta \text{Speaker Similarity} \ge +0.02$ compared to the Zero-Shot baseline.
   - Candidate WER does not significantly degrade relative to Zero-Shot.

If any criterion fails, the candidate is **REJECTED**, Zero-Shot remains primary, and the system recommends collecting an additional batch of recordings.

---

## 8. Adaptive Retraining Workflow
When a candidate is rejected:
1. Active voice generation remains Zero-Shot XTTS.
2. The user continues normal interaction, collecting additional recordings.
3. At $+25\text{--}50$ new valid recordings (e.g. 75 total recordings for Round 2), the pipeline triggers retraining.
4. Retraining incorporates the new data while preserving the original validation set for fair evaluation comparison.

---

## 9. Inference & Zero-Shot Fallback
At runtime:
```text
User Request (user_id, text)
          ↓
PersonalizedVoiceManager
          ↓
Is fine-tuned model promoted for user?
  ├── YES → Attempt Fine-Tuned XTTS Inference
  │           ├── Success → Return generated audio
  │           └── Failure (e.g. OOM) → [Fallback] → Zero-Shot XTTS
  └── NO  → Execute Zero-Shot XTTS
```

---

## 10. How to Run the Pipeline

### Check Readiness / Run Adaptive Fine-Tuning
```python
from Voice_Finetuning.pipeline import AdaptiveFineTuningPipeline

pipeline = AdaptiveFineTuningPipeline()
result = pipeline.run_for_user("user_001")

print(f"Readiness: {result.data_status.readiness_status.value}")
print(f"Valid recordings: {result.data_status.valid_recordings_count}/{result.data_status.required_threshold}")
print(f"Decision: {result.decision}")
```

### Personalized Voice Synthesis
```python
from Voice_Finetuning.personalized_voice_manager import PersonalizedVoiceManager

pvm = PersonalizedVoiceManager()
output_path = pvm.generate_for_user(
    user_id="user_001",
    text="Hello, I am your personalized assistant."
)
print(f"Generated speech: {output_path}")
```

---

## 11. Inspecting Results
Historical records, metrics, and CSV summaries are stored under:
- `Voice_Finetuning/results/{user_id}/evaluation_results.csv`
- `Voice_Finetuning/checkpoints/{user_id}/history.json`
- `Voice_Finetuning/checkpoints/{user_id}/round_{n}/metadata.json`

---

## 12. Rollback to Zero-Shot
To instantly revert a user's voice model to Zero-Shot XTTS:
```python
from Voice_Finetuning.checkpoint_manager import CheckpointManager

chk_mgr = CheckpointManager()
chk_mgr.rollback_to_zero_shot("user_001")
print("Reverted user_001 to Zero-Shot XTTS.")
```
