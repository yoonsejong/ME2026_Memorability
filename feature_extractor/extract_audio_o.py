import sys
import numpy as np
import psutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent))
from audio_o import extract_audio_features, MAX_DURATION

_VIDEO_EXTS = (".mp4", ".mkv", ".webm")

def _find_video(video_dir: Path, vid: str):
    for ext in _VIDEO_EXTS:
        p = video_dir / f"{vid}{ext}"
        if p.exists():
            return p
    return None

FEATURE_KEYS = [
    "energy_mean", "energy_std", "pitch_mean_hz", "pitch_std_hz",
    "pitch_variation", "snr_db", "speech_ratio", "syllable_count", "speaking_rate"
]


def _process(args):
    vid, video_path, transcript_path, out_dir, max_duration = args
    out_dir = Path(out_dir)
    if (out_dir / f"{vid}.npy").exists():
        return "skip", vid, None
    if video_path is None:
        return "skip", vid, None
    try:
        feat_dict = extract_audio_features(str(video_path), transcript_path, max_duration=max_duration)
        features = np.array([feat_dict[k] for k in FEATURE_KEYS])
        np.save(out_dir / f"{vid}.npy", features)
        msg = f"snr_db={feat_dict['snr_db']:.2f}  speaking_rate={feat_dict['speaking_rate']:.2f}"
        return "ok", vid, msg
    except Exception as e:
        return "fail", vid, str(e)


def run(video_ids, video_dir, stt_dir, out_dir, max_duration=MAX_DURATION):
    video_dir, stt_dir, out_dir = Path(video_dir), Path(stt_dir), Path(out_dir)
    total = len(video_ids)
    done = skipped = failed = 0

    args = []
    for vid in video_ids:
        txt = stt_dir / f"{vid}.txt"
        args.append((vid, _find_video(video_dir, vid),
                     str(txt) if txt.exists() else None,
                     str(out_dir), max_duration))

    with ProcessPoolExecutor(max_workers=max(1, psutil.cpu_count(logical=False)-1)) as pool:
        futures = {pool.submit(_process, a): a[0] for a in args}
        for fut in as_completed(futures):
            status, vid, msg = fut.result()
            if status == "ok":
                print(f"[ok]   {vid}  {msg}")
                done += 1
            elif status == "skip":
                print(f"[skip] {vid}")
                skipped += 1
            else:
                print(f"[fail] {vid}  {msg}")
                failed += 1

    print(f"\ndone={done}/{total}  skipped={skipped}  failed={failed}")


if __name__ == "__main__":
    print('Please run through test_all.py')
