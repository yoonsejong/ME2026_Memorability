"""
check_audio_output.py - Validate extracted audio .npy feature files for anomalies.

Checks per file:
  - Load error
  - Wrong shape
  - Inf values
  - Unexpected NaN (audio_o: indices 7-8 are NaN when no transcript — expected)
  - All-zero valid elements: verified via ffmpeg volumedetect
      benign if max_volume < -50 dB, FAIL otherwise

Called from get_all_features.py. Not intended to be run standalone.
"""

import re
import subprocess
import numpy as np
from pathlib import Path

# audio_o indices 7 (syllable_count) and 8 (speaking_rate) are NaN when no transcript
AUDIO_O_OPTIONAL_NAN = frozenset({7, 8})
SILENCE_THRESHOLD_DB = -50

_VIDEO_EXTS = (".mp4", ".mkv", ".webm")

def _find_video(video_dir, vid: str):
    for ext in _VIDEO_EXTS:
        p = Path(video_dir) / f"{vid}{ext}"
        if p.exists():
            return p
    return None


def _max_volume_db(video_path):
    """Return max volume in dB via ffmpeg volumedetect, or None on error."""
    result = subprocess.run(
        ["ffmpeg", "-i", str(video_path), "-af", "volumedetect", "-vn", "-f", "null", "-"],
        capture_output=True, text=True
    )
    for line in result.stderr.splitlines():
        m = re.search(r"max_volume:\s*([-\d.]+)\s*dB", line)
        if m:
            return float(m.group(1))
    return None


def check_file(path, expected_shape, kind, video_dir):
    vid = path.stem
    issues = []

    try:
        arr = np.load(path)
    except Exception as e:
        return [f"FAIL load error: {e}"]

    if arr.shape != expected_shape:
        return [f"FAIL wrong shape {arr.shape}, expected {expected_shape}"]

    flat = arr.flatten()
    n = len(flat)

    inf_n = int(np.sum(np.isinf(flat)))
    if inf_n:
        issues.append(f"FAIL {inf_n}/{n} Inf")

    nan_mask = np.isnan(flat)
    nan_n = int(np.sum(nan_mask))

    if kind == "audio_o":
        unexpected_nan = [i for i in range(n) if nan_mask[i] and i not in AUDIO_O_OPTIONAL_NAN]
        if unexpected_nan:
            issues.append(f"FAIL NaN at unexpected indices {unexpected_nan}")
    else:
        if nan_n == n:
            issues.append("FAIL all NaN")
        elif nan_n:
            issues.append(f"FAIL {nan_n}/{n} NaN")

    valid = flat[np.isfinite(flat)]
    if len(valid) > 0 and np.all(valid == 0):
        video_path = _find_video(video_dir, vid)
        vol = _max_volume_db(video_path) if video_path is not None else None
        if vol is not None and vol < SILENCE_THRESHOLD_DB:
            issues.append(f"BENIGN silent audio (max_volume={vol:.1f} dB)")
        elif vol is not None:
            issues.append(f"FAIL all-zero but audio not silent (max_volume={vol:.1f} dB)")
        else:
            issues.append("FAIL all-zero and could not determine audio level")

    return issues


def run(dirs):
    """Check audio output dirs. Returns (lines, fail_count).

    Args:
        dirs: list of (out_dir, expected_shape, kind, video_dir)
    """
    lines = []
    total = ok = warn_total = fail_total = 0

    for out_dir, expected_shape, kind, video_dir in dirs:
        d = Path(out_dir)
        if not d.exists():
            lines.append(f"[skip] {d.name}/ not found")
            continue

        files = sorted(d.glob("*.npy"))
        if not files:
            lines.append(f"[skip] {d.name}/ empty")
            continue

        dir_ok = dir_warn = dir_fail = 0
        flagged = []

        for path in files:
            issues = check_file(path, expected_shape, kind, video_dir)
            is_fail = any(i.startswith("FAIL") for i in issues)
            is_benign = any(i.startswith("BENIGN") for i in issues)
            if is_fail:
                dir_fail += 1
                flagged.append((path.stem, issues))
            elif is_benign:
                dir_warn += 1
                flagged.append((path.stem, issues))
            else:
                dir_ok += 1

        n = len(files)
        status = "ok  " if not flagged else ("FAIL" if dir_fail else "warn")
        lines.append(f"[{status}] {d.name:<28} {n:>4} files  ok={dir_ok}  warn={dir_warn}  fail={dir_fail}")
        for vid, issues in flagged:
            for issue in issues:
                lines.append(f"         {vid}: {issue}")

        total += n
        ok += dir_ok
        warn_total += dir_warn
        fail_total += dir_fail

    lines.append(f"\n{'='*60}")
    lines.append(f"total {total} files — ok={ok}  warn={warn_total}  fail={fail_total}")

    return lines, fail_total


if __name__ == "__main__":
    print("Please run through get_all_features.py")
