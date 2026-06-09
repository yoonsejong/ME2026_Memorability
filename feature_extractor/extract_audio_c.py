import sys
import numpy as np
import psutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent))
from audio_c import extract_audio_features, MAX_DURATION

_VIDEO_EXTS = (".mp4", ".mkv", ".webm")

def _find_video(video_dir: Path, vid: str):
    for ext in _VIDEO_EXTS:
        p = video_dir / f"{vid}{ext}"
        if p.exists():
            return p
    return None


def _process(args):
    vid, video_path, out_dir, max_duration = args
    out_dir = Path(out_dir)
    if (out_dir / f"{vid}.npy").exists():
        return "skip", vid, None
    if video_path is None:
        return "skip", vid, None
    try:
        features = extract_audio_features(str(video_path), max_duration=max_duration)
        np.save(out_dir / f"{vid}.npy", features)
        return "ok", vid, f"shape={features.shape}"
    except Exception as e:
        return "fail", vid, str(e)


def run(video_ids, video_dir, out_dir, max_duration=MAX_DURATION):
    video_dir, out_dir = Path(video_dir), Path(out_dir)
    total = len(video_ids)
    done = skipped = failed = 0

    args = [(vid, _find_video(video_dir, vid), str(out_dir), max_duration)
            for vid in video_ids]

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
