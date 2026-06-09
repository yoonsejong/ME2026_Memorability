import sys
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent))
from emotion import extract_emotion_features_array, extract_emotion_features


def _process(args):
    vid, txt_path, out_dir = args
    txt, out_dir = Path(txt_path), Path(out_dir)
    if (out_dir / f"{vid}.npy").exists():
        return "skip", vid, None
    if not txt.exists():
        return "skip", vid, None
    try:
        text = txt.read_text(encoding="utf-8")
        info = extract_emotion_features(text)
        features = extract_emotion_features_array(text, precomputed=info)
        np.save(out_dir / f"{vid}.npy", features)
        return "ok", vid, f"shape={features.shape}  dominant={info['dominant_emotion']}"
    except Exception as e:
        return "fail", vid, str(e)


def run(video_ids, stt_dir, out_dir):
    stt_dir, out_dir = Path(stt_dir), Path(out_dir)
    total = len(video_ids)
    done = skipped = failed = 0

    args = [(vid, str(stt_dir / f"{vid}.txt"), str(out_dir)) for vid in video_ids]

    with ProcessPoolExecutor() as pool:
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
