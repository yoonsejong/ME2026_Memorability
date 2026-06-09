"""
audio_o.py - Extract 9 acoustic features (Olufela's approach)

Features:
  - energy_mean, energy_std
  - pitch_mean_hz, pitch_std_hz, pitch_variation
  - snr_db, speech_ratio
  - syllable_count, speaking_rate
"""

import subprocess
import os
import tempfile
import librosa
import numpy as np
import re
from pathlib import Path


SAMPLE_RATE = 16000
MAX_DURATION = 120


def count_syllables(text: str) -> int:
    """Estimate syllable count using vowel groups."""
    text = re.sub(r"[^a-z ]", "", text.lower())
    total = 0
    for word in text.split():
        total += max(1, len(re.findall(r"[aeiou]+", word)))
    return total


def extract_audio_features(filepath: str, transcript_path: str = None, max_duration: int = MAX_DURATION) -> dict:
    """Extract 9 acoustic features from audio file.

    Args:
        filepath: Path to audio/video file (any ffmpeg-readable format)
        transcript_path: Path to transcript text file (optional)
    
    Returns:
        Dictionary with 9 acoustic features
    """
    fd, tmp_wav = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(filepath), "-ar", str(SAMPLE_RATE), "-ac", "1",
             "-vn", "-t", str(max_duration), tmp_wav],
            capture_output=True, check=True
        )
        y, sr = librosa.load(tmp_wav, sr=SAMPLE_RATE)
    finally:
        os.unlink(tmp_wav)

    hop_length = 512

    # Energy
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    energy_mean = float(np.mean(rms))
    energy_std = float(np.std(rms))

    # Pitch using pYIN
    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
        hop_length=hop_length
    )
    voiced_f0 = f0[voiced_flag]

    if len(voiced_f0) > 0:
        pitch_mean = float(np.mean(voiced_f0))
        pitch_std = float(np.std(voiced_f0))
        mid = len(voiced_f0) // 2
        pitch_variation = float(np.mean(voiced_f0[mid:]) - np.mean(voiced_f0[:mid]))
    else:
        pitch_mean = pitch_std = pitch_variation = 0.0

    # SNR
    threshold = np.percentile(rms, 35)
    speech_frames = rms > threshold
    noise_rms = rms[~speech_frames]
    speech_ratio = float(np.mean(speech_frames))

    noise_power = np.mean(noise_rms ** 2)
    if noise_power > 0:
        speech_rms = rms[speech_frames]
        snr_db = float(10 * np.log10(np.mean(speech_rms ** 2) / noise_power))
    else:
        snr_db = 60.0

    # Speaking rate
    speech_duration = float(np.sum(speech_frames) * hop_length / sr)

    if transcript_path and Path(transcript_path).exists():
        with open(transcript_path, "r", encoding="utf-8") as f:
            text = f.read()
        syllables = count_syllables(text)
        speaking_rate = syllables / speech_duration if speech_duration > 0 else float("nan")
    else:
        syllables = float("nan")
        speaking_rate = float("nan")

    return {
        "energy_mean": energy_mean,
        "energy_std": energy_std,
        "pitch_mean_hz": pitch_mean,
        "pitch_std_hz": pitch_std,
        "pitch_variation": pitch_variation,
        "snr_db": snr_db,
        "speech_ratio": speech_ratio,
        "syllable_count": syllables,
        "speaking_rate": speaking_rate,
    }


if __name__ == "__main__":
    import sys
    import soundfile as sf

    sr = SAMPLE_RATE
    n = sr * 3
    t = np.linspace(0, 3, n, endpoint=False, dtype=np.float32)

    results = [0, 0]  # [passed, failed]

    def check(name, condition, detail=""):
        if condition:
            print(f"[pass] {name}")
            results[0] += 1
        else:
            print(f"[FAIL] {name}" + (f"  {detail}" if detail else ""))
            results[1] += 1

    # --- count_syllables ---
    check("syllables: 'hello world' = 3", count_syllables("hello world") == 3)
    check("syllables: 'rhythm' = 1",      count_syllables("rhythm") == 1)
    check("syllables: '' = 0",            count_syllables("") == 0)

    # --- extract_audio_features ---
    y_silent = np.zeros(n, dtype=np.float32)
    y_tone   = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    fd, tmp_wav = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    fd, tmp_txt = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    try:
        # silent
        sf.write(tmp_wav, y_silent, sr)
        feat = extract_audio_features(tmp_wav)
        check("silent: energy_mean ~0",    feat["energy_mean"] < 1e-6, str(feat["energy_mean"]))
        check("silent: pitch_mean_hz = 0", feat["pitch_mean_hz"] == 0.0)

        # constant 220 Hz tone
        sf.write(tmp_wav, y_tone, sr)
        feat = extract_audio_features(tmp_wav)
        check("tone: pitch_mean_hz ~220",  abs(feat["pitch_mean_hz"] - 220) < 5, f"{feat['pitch_mean_hz']:.1f}")
        check("tone: energy_mean > 0",     feat["energy_mean"] > 0)
        check("tone: snr_db finite",       np.isfinite(feat["snr_db"]), str(feat["snr_db"]))

        # transcript → syllable count and speaking rate
        with open(tmp_txt, "w", encoding="utf-8") as fh:
            fh.write("hello world")
        feat = extract_audio_features(tmp_wav, transcript_path=tmp_txt)
        check("transcript: syllable_count = 3", feat["syllable_count"] == 3,  str(feat["syllable_count"]))
        check("transcript: speaking_rate > 0",  feat["speaking_rate"] > 0,    str(feat["speaking_rate"]))
    finally:
        os.unlink(tmp_wav)
        os.unlink(tmp_txt)

    print(f"\n{results[0]} passed, {results[1]} failed")
    sys.exit(0 if results[1] == 0 else 1)
