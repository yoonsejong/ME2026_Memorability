"""
check_text_output.py - Validate extracted text .npy feature files for anomalies.

Checks per file:
  - Load error
  - Wrong shape
  - Inf values
  - All-NaN
  - Partial NaN or all-zero: verified against transcript file
      benign if transcript is empty, FAIL if transcript has content

Called from get_all_features.py. Not intended to be run standalone.
"""

import numpy as np
from pathlib import Path


def _is_empty_transcript(txt_path):
    if not txt_path.exists():
        return True
    return len(txt_path.read_text(encoding="utf-8").strip()) == 0


def check_file(path, expected_shape, stt_dir):
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

    nan_n = int(np.sum(np.isnan(flat)))
    valid = flat[np.isfinite(flat)]
    all_zero = len(valid) > 0 and np.all(valid == 0)

    if nan_n == n:
        issues.append("FAIL all NaN")
    elif nan_n or all_zero:
        txt_path = Path(stt_dir) / f"{vid}.txt"
        if _is_empty_transcript(txt_path):
            parts = []
            if nan_n:
                parts.append(f"{nan_n}/{n} NaN")
            if all_zero:
                parts.append("all-zero valid elements")
            issues.append(f"BENIGN empty transcript ({', '.join(parts)})")
        else:
            if nan_n:
                issues.append(f"FAIL {nan_n}/{n} NaN (transcript has content)")
            if all_zero:
                issues.append("FAIL all-zero valid elements (transcript has content)")

    return issues


def run(dirs):
    """Check text output dirs. Returns (lines, fail_count).

    Args:
        dirs: list of (out_dir, expected_shape, stt_dir)
    """
    lines = []
    total = ok = warn_total = fail_total = 0

    for out_dir, expected_shape, stt_dir in dirs:
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
            issues = check_file(path, expected_shape, stt_dir)
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
