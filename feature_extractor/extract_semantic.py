import sys
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent))
from semantic import build_word_frequency, extract_text_features

_word_freq = None


def _init_worker(word_freq):
    global _word_freq
    _word_freq = word_freq


def _process(args):
    vid, txt_path, out_dir = args
    txt, out_dir = Path(txt_path), Path(out_dir)
    if not txt.exists():
        return "skip", vid, None
    try:
        text = txt.read_text(encoding="utf-8")
        features = extract_text_features(text, _word_freq)
        np.save(out_dir / f"{vid}.npy", features)
        return "ok", vid, f"shape={features.shape}"
    except Exception as e:
        return "fail", vid, str(e)


def run(video_ids, stt_dir, out_dir, vocab_dirs=None):
    stt_dir, out_dir = Path(stt_dir), Path(out_dir)

    pending = [vid for vid in video_ids if not (out_dir / f"{vid}.npy").exists()]
    if not pending:
        print(f"all {len(video_ids)} already done, skipping word frequency build")
        return

    dirs = vocab_dirs if vocab_dirs is not None else stt_dir
    word_freq = build_word_frequency(dirs)
    print(f"vocab size: {len(word_freq)}")

    total = len(pending)
    done = skipped = failed = 0

    args = [(vid, str(stt_dir / f"{vid}.txt"), str(out_dir)) for vid in pending]

    with ProcessPoolExecutor(initializer=_init_worker, initargs=(word_freq,)) as pool:
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
