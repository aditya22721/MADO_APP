"""
Utility functions for Voice_Finetuning subsystem.

Provides:
    - Text normalization and cleaning
    - Levenshtein-based Word Error Rate (WER) & Character Error Rate (CER)
    - Normalized text similarity
    - Audio signal quality & stability analysis
    - Cryptographic audio file hashing for deduplication
"""

import hashlib
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import soundfile as sf
import torch


# ============================================================
# TEXT UTILITIES
# ============================================================

def normalize_text(text: str) -> str:
    """
    Normalize text for objective error rate calculation and comparison.
    Converts to lowercase, removes punctuation, and collapses multiple spaces.
    """
    if not text or not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_levenshtein_distance(seq1: List[str], seq2: List[str]) -> int:
    """Compute standard Levenshtein edit distance between two sequences."""
    n, m = len(seq1), len(seq2)
    if n == 0:
        return m
    if m == 0:
        return n

    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,        # deletion
                d[i][j - 1] + 1,        # insertion
                d[i - 1][j - 1] + cost  # substitution
            )

    return d[n][m]


def compute_wer(reference: str, hypothesis: str) -> float:
    """
    Compute Word Error Rate (WER) between reference and hypothesis text.
    WER = edit_distance(words) / len(reference_words).
    Capped at 1.0 for completely inaccurate / degenerate output.
    """
    ref_norm = normalize_text(reference)
    hyp_norm = normalize_text(hypothesis)

    ref_words = ref_norm.split()
    hyp_words = hyp_norm.split()

    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    distance = compute_levenshtein_distance(ref_words, hyp_words)
    wer = distance / len(ref_words)
    return float(min(1.0, wer))


def compute_cer(reference: str, hypothesis: str) -> float:
    """Compute Character Error Rate (CER)."""
    ref_norm = normalize_text(reference)
    hyp_norm = normalize_text(hypothesis)

    ref_chars = list(ref_norm.replace(" ", ""))
    hyp_chars = list(hyp_norm.replace(" ", ""))

    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0

    distance = compute_levenshtein_distance(ref_chars, hyp_chars)
    cer = distance / len(ref_chars)
    return float(min(1.0, cer))


def compute_text_similarity(reference: str, hypothesis: str) -> float:
    """
    Compute normalized text similarity in range [0.0, 1.0].
    Similarity = 1.0 - WER.
    """
    return max(0.0, 1.0 - compute_wer(reference, hypothesis))


# ============================================================
# AUDIO SIGNAL UTILITIES
# ============================================================

def compute_file_hash(file_path: Union[str, Path]) -> str:
    """Compute SHA-256 checksum of an audio file for deduplication."""
    file_path = Path(file_path)
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def analyze_audio_signal(
    audio_path: Union[str, Path],
    silence_threshold: float = 0.01,
    clipping_threshold: float = 0.999
) -> Dict[str, Union[float, bool, int]]:
    """
    Perform deep signal quality analysis on an audio file.

    Returns:
        Dict containing:
            - duration: Total duration in seconds
            - sample_rate: Audio sample rate
            - channels: Channel count
            - peak_amplitude: Peak absolute amplitude
            - rms_energy: Root mean square energy
            - clipping_ratio: Proportion of samples at clipping threshold
            - silence_ratio: Proportion of samples below silence threshold
            - has_nan_inf: Boolean indicating invalid floating values
            - is_empty: Boolean indicating zero samples
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    audio_data, sample_rate = sf.read(str(audio_path), dtype="float32")

    if audio_data.size == 0:
        return {
            "duration": 0.0,
            "sample_rate": sample_rate,
            "channels": 0,
            "peak_amplitude": 0.0,
            "rms_energy": 0.0,
            "clipping_ratio": 0.0,
            "silence_ratio": 1.0,
            "has_nan_inf": False,
            "is_empty": True,
        }

    # Convert to 1D array if stereo/multichannel
    if audio_data.ndim > 1:
        channels = audio_data.shape[1]
        mono_data = np.mean(audio_data, axis=1)
    else:
        channels = 1
        mono_data = audio_data

    total_samples = len(mono_data)
    duration = total_samples / sample_rate

    has_nan_inf = bool(np.isnan(mono_data).any() or np.isinf(mono_data).any())
    if has_nan_inf:
        mono_data = np.nan_to_num(mono_data, nan=0.0, posinf=0.0, neginf=0.0)

    abs_data = np.abs(mono_data)
    peak_amplitude = float(np.max(abs_data)) if total_samples > 0 else 0.0
    rms_energy = float(np.sqrt(np.mean(mono_data ** 2))) if total_samples > 0 else 0.0

    clipped_samples = np.sum(abs_data >= clipping_threshold)
    clipping_ratio = float(clipped_samples / total_samples) if total_samples > 0 else 0.0

    effective_silence_threshold = max(0.005, 0.05 * peak_amplitude) if peak_amplitude > 0 else silence_threshold
    silent_samples = np.sum(abs_data <= effective_silence_threshold)
    silence_ratio = float(silent_samples / total_samples) if total_samples > 0 else 0.0

    return {
        "duration": float(duration),
        "sample_rate": int(sample_rate),
        "channels": int(channels),
        "peak_amplitude": peak_amplitude,
        "rms_energy": rms_energy,
        "clipping_ratio": clipping_ratio,
        "silence_ratio": silence_ratio,
        "has_nan_inf": has_nan_inf,
        "is_empty": False,
    }
