"""
audio_c.py - Extract 9 audio features (Conner's approach)

Features:
  - pitch_mean, pitch_var, pitch_range
  - pitch_vel_mean, pitch_vel_var
  - energy_mean, energy_var, energy_range
  - pause_var
"""

import subprocess
import json
import os
import tempfile
import librosa
import numpy as np

MAX_DURATION = 60


def _get_duration(filepath):
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(filepath)],
        capture_output=True, text=True, check=True
    )
    return float(json.loads(r.stdout)["format"]["duration"])


def extract_audio_features_from_array(y, sr):
    """Extract 9 audio features from audio array."""
    hop_length = 512
    frame_duration = hop_length / sr

    # Energy
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]

    energy_mean = np.mean(rms)
    energy_var = np.var(rms)
    energy_range = np.max(rms) - np.min(rms)

    # Pauses
    intervals = librosa.effects.split(y, top_db=30)

    pauses = [(b[0] - a[1]) / sr for a, b in zip(intervals, intervals[1:])]

    pause_var = np.var(pauses) if pauses else 0

    # Pitch
    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        sr=sr,
        hop_length=hop_length
    )

    voiced_idx = np.where(voiced_flag)[0]
    voiced_f0  = f0[voiced_idx]

    if len(voiced_f0) >= 1:
        pitch_mean  = np.mean(voiced_f0)
        pitch_var   = np.var(voiced_f0)
        pitch_range = np.max(voiced_f0) - np.min(voiced_f0)
    else:
        pitch_mean = pitch_var = pitch_range = 0.0

    if len(voiced_f0) >= 2:
        actual_dt      = np.diff(voiced_idx) * frame_duration
        pitch_velocity = np.diff(voiced_f0) / actual_dt
        pitch_vel_mean = np.mean(np.abs(pitch_velocity))
        pitch_vel_var  = np.var(pitch_velocity)
    else:
        pitch_vel_mean = pitch_vel_var = 0.0

    return np.array([
        pitch_mean,
        pitch_var,
        pitch_range,
        pitch_vel_mean,
        pitch_vel_var,
        energy_mean,
        energy_var,
        energy_range,
        pause_var
    ])


def extract_audio_features(filepath, max_duration=MAX_DURATION):
    """Extract 9 audio features from start, middle, and end segments.

    Each segment is up to max_duration seconds. If the video is shorter
    than 3 * max_duration, the video is split evenly into three parts.

    Returns:
        np.array of shape (3, 9) - one row per segment
    """
    total = _get_duration(filepath)

    if total >= 3 * max_duration:
        segments = [
            (0,                              max_duration),
            (total / 2 - max_duration / 2,   max_duration),
            (total - max_duration,            max_duration),
        ]
    else:
        seg_dur = total / 3
        segments = [
            (0,           seg_dur),
            (seg_dur,     seg_dur),
            (2 * seg_dur, seg_dur),
        ]

    features = []
    for offset, dur in segments:
        fd, tmp_wav = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-ss", str(offset), "-t", str(dur),
                 "-i", str(filepath), "-ar", "16000", "-ac", "1", "-vn", tmp_wav],
                capture_output=True, check=True
            )
            y, sr = librosa.load(tmp_wav, sr=16000)
            features.append(extract_audio_features_from_array(y, sr))
        finally:
            os.unlink(tmp_wav)

    return np.array(features)


if __name__ == "__main__":
    import sys
    import soundfile as sf

    sr = 16000
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

    # --- extract_audio_features_from_array ---

    # silent: all features should be zero
    y_silent = np.zeros(n, dtype=np.float32)
    f = extract_audio_features_from_array(y_silent, sr)
    check("silent: shape (9,)", f.shape == (9,))
    check("silent: all zeros", np.all(f == 0.0), str(f))

    # constant 220 Hz tone
    y_tone = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    f = extract_audio_features_from_array(y_tone, sr)
    check("tone: pitch_mean ~220 Hz", abs(f[0] - 220) < 5, f"pitch_mean={f[0]:.1f}")
    check("tone: pitch_var low", f[1] < 50, f"pitch_var={f[1]:.1f}")
    check("tone: energy_mean > 0", f[5] > 0, f"energy_mean={f[5]:.5f}")

    # chirp 200→400 Hz: pitch should vary and have non-zero velocity
    phase = 2 * np.pi * (200 * t + 50 * t ** 2)
    y_chirp = (0.5 * np.sin(phase)).astype(np.float32)
    f = extract_audio_features_from_array(y_chirp, sr)
    check("chirp: pitch_var > 0", f[1] > 0, f"pitch_var={f[1]:.1f}")
    check("chirp: pitch_vel_mean > 0", f[3] > 0, f"pitch_vel_mean={f[3]:.2f}")

    # --- extract_audio_features (file-based, requires ffprobe) ---
    fd, tmp_wav = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        sf.write(tmp_wav, y_tone, sr)
        feat_3seg = extract_audio_features(tmp_wav)
        check("file: shape (3, 9)", feat_3seg.shape == (3, 9), str(feat_3seg.shape))
        check("file: pitch_mean > 0 in all segments",
              np.all(feat_3seg[:, 0] > 0), str(feat_3seg[:, 0]))
    finally:
        os.unlink(tmp_wav)

    print(f"\n{results[0]} passed, {results[1]} failed")
    sys.exit(0 if results[1] == 0 else 1)
