import sys
import csv
import zipfile
from pathlib import Path

WORK_DIR = Path(__file__).parent

# --- devset ---
DEV_CSV       = WORK_DIR / "../devset/devset_videolist_GT.csv"
DEV_VIDEO_DIR = WORK_DIR / "../video-devset"
DEV_STT_DIR   = WORK_DIR / "../devset/devset-stt"

DEV_OUT_AUDIO_C  = WORK_DIR / "extracted_audio_c_dev"
DEV_OUT_AUDIO_O  = WORK_DIR / "extracted_audio_o_dev"
DEV_OUT_SEMANTIC = WORK_DIR / "extracted_semantic_dev"
DEV_OUT_EMOTION  = WORK_DIR / "extracted_emotion_dev"

# --- testset ---
TEST_CSV       = WORK_DIR / "../testset/testset_videolist_.csv"
TEST_VIDEO_DIR = WORK_DIR / "../video-testset"
TEST_STT_DIR   = WORK_DIR / "../testset/testset-stt"

TEST_OUT_AUDIO_C  = WORK_DIR / "extracted_audio_c_test"
TEST_OUT_AUDIO_O  = WORK_DIR / "extracted_audio_o_test"
TEST_OUT_SEMANTIC = WORK_DIR / "extracted_semantic_test"
TEST_OUT_EMOTION  = WORK_DIR / "extracted_emotion_test"

for _d in [DEV_OUT_AUDIO_C, DEV_OUT_AUDIO_O, DEV_OUT_SEMANTIC, DEV_OUT_EMOTION,
           TEST_OUT_AUDIO_C, TEST_OUT_AUDIO_O, TEST_OUT_SEMANTIC, TEST_OUT_EMOTION]:
    _d.mkdir(exist_ok=True)

# --- output zip ---
OUTPUT_ZIP = WORK_DIR / "features.zip"

# --- audio duration parameters ---
MAX_DURATION_AUDIO_O = 120  # seconds of audio loaded for global acoustic features
MAX_DURATION_AUDIO_C = 60   # seconds per segment (start / middle / end) for temporal features

sys.path.insert(0, str(WORK_DIR))
import extract_audio_c
import extract_audio_o
import extract_semantic
import extract_emotion
import check_audio_output
import check_text_output


def run_all(csv_path, video_dir, stt_dir, out_audio_c, out_audio_o, out_semantic, out_emotion, vocab_dirs=None):
    with open(csv_path, newline="", encoding="utf-8") as f:
        video_ids = [row["id"] for row in csv.DictReader(f)]

    print(f"=== emotion ({len(video_ids)} videos) ===")
    extract_emotion.run(video_ids, stt_dir, out_emotion)

    print(f"\n=== semantic ({len(video_ids)} videos) ===")
    extract_semantic.run(video_ids, stt_dir, out_semantic, vocab_dirs=vocab_dirs)

    print(f"\n=== audio_o ({len(video_ids)} videos) ===")
    extract_audio_o.run(video_ids, video_dir, stt_dir, out_audio_o, max_duration=MAX_DURATION_AUDIO_O)

    print(f"\n=== audio_c ({len(video_ids)} videos) ===")
    extract_audio_c.run(video_ids, video_dir, out_audio_c, max_duration=MAX_DURATION_AUDIO_C)


def check_outputs():
    """Run audio and text output checks. Returns (report_str, fail_count)."""
    audio_dirs = [
        (DEV_OUT_AUDIO_C,  (3, 9), "audio_c", DEV_VIDEO_DIR),
        (TEST_OUT_AUDIO_C, (3, 9), "audio_c", TEST_VIDEO_DIR),
        (DEV_OUT_AUDIO_O,  (9,),   "audio_o", DEV_VIDEO_DIR),
        (TEST_OUT_AUDIO_O, (9,),   "audio_o", TEST_VIDEO_DIR),
    ]
    text_dirs = [
        (DEV_OUT_SEMANTIC,  (7,),  DEV_STT_DIR),
        (TEST_OUT_SEMANTIC, (7,),  TEST_STT_DIR),
        (DEV_OUT_EMOTION,   (19,), DEV_STT_DIR),
        (TEST_OUT_EMOTION,  (19,), TEST_STT_DIR),
    ]

    audio_lines, audio_fails = check_audio_output.run(audio_dirs)
    text_lines, text_fails = check_text_output.run(text_dirs)

    all_lines = (
        ["=== audio ==="] + audio_lines +
        ["", "=== text ==="] + text_lines
    )
    for line in all_lines:
        print(line)

    return "\n".join(all_lines), audio_fails + text_fails


def zip_outputs(check_report):
    dirs = [
        DEV_OUT_AUDIO_C, DEV_OUT_AUDIO_O, DEV_OUT_SEMANTIC, DEV_OUT_EMOTION,
        TEST_OUT_AUDIO_C, TEST_OUT_AUDIO_O, TEST_OUT_SEMANTIC, TEST_OUT_EMOTION,
    ]
    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for d in dirs:
            for f in sorted(d.rglob("*.npy")):
                zf.write(f, f.relative_to(WORK_DIR))
        zf.writestr("check_report.txt", check_report)
    print(f"zipped: {OUTPUT_ZIP}")


if __name__ == "__main__":
    print("====== devset ======")
    run_all(DEV_CSV, DEV_VIDEO_DIR, DEV_STT_DIR,
            DEV_OUT_AUDIO_C, DEV_OUT_AUDIO_O, DEV_OUT_SEMANTIC, DEV_OUT_EMOTION)

    print("\n====== testset ======")
    run_all(TEST_CSV, TEST_VIDEO_DIR, TEST_STT_DIR,
            TEST_OUT_AUDIO_C, TEST_OUT_AUDIO_O, TEST_OUT_SEMANTIC, TEST_OUT_EMOTION,
            vocab_dirs=[DEV_STT_DIR, TEST_STT_DIR])

    print("\n====== checking outputs ======")
    report, fail_total = check_outputs()

    if fail_total > 0:
        print(f"\n{fail_total} failure(s) found — skipping zip")
    else:
        print("\n====== zipping outputs ======")
        zip_outputs(report)
