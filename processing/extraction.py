import argparse
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from huggingface_hub import hf_hub_download

ALL_EFFECTS = ["cloud","fog","rain","snow","puddle","snowflake"]

DEFAULT_OUT = Path("/Users/jiadonggu/rainyday_v1/data/extracted")

#helpers
def have_ffmpeg():
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None

def get_frame_count(video_path):
    result = subprocess.run(
         [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-count_frames", "-show_entries", "stream=nb_read_frames",
            "-of", "default=nokey=1:noprint_wrappers=1", str(video_path),
        ],
        capture_output = True,
        text = True,
        check = True,
    )
    return int(result.stdout.strip())

def extract_frames(video_path,frame_indices):
    out_jpgs = []
    for idx in frame_indices:
        with tempfile.NamedTemporaryFile(suffix = ".jpg", delete = False) as t:
            tmp = Path(t.name)
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-i", str(video_path),
                    "-vf", f"select=eq(n\\,{idx})",
                    "-vframes", "1",
                    "-q:v", "2",
                    str(tmp),
                ],
                check=True,
            )
            out_jpgs.append(tmp.read_bytes())
        finally:
            tmp.unlink(missing_ok=True)

    return out_jpgs

def process_one_tar(
        tar_path, effect, tar_idx, out_dir, frames_per_video
):
    n_processed = 0
    n_skipped = 0
    with tarfile.open(tar_path) as tf:
        sample_ids = sorted({
            m.name.split("/")[1]
            for m in tf.getmembers()
            if m.name.count("/") >= 2 and m.name.startswith("./")
        })

        for sample_id in sample_ids:
            stem = f"{tar_idx:03d}_{sample_id}"
            label_path = out_dir / f"{stem}.json"

            if label_path.exists():
                n_skipped += 1
                continue
            try:
                meta_m = tf.getmember(f"./{sample_id}/0000.meta.json")
                video_m = tf.getmember(f"./{sample_id}/video.mp4")
            except KeyError: #some dirs dont exist
                continue

            meta = json.load(tf.extractfile(meta_m))

            with tempfile.NamedTemporaryFile(suffix = ".mp4", delete = False) as t:
                video_tmp = Path(t.name)
                video_tmp.write_bytes(tf.extractfile(video_m).read())
            try:
                n_frames = get_frame_count(video_tmp)
                if n_frames < frames_per_video:
                    print(f"WARNING: sample {sample_id} has only {n_frames} frames")
                    continue

                indices = [
                    round(i*(n_frames-1) / (frames_per_video - 1))
                    for i in range(frames_per_video)
                ]
                jpgs = extract_frames(video_tmp, indices)
            finally:
                video_tmp.unlink(missing_ok = True)

            (out_dir / f"{stem}_clear.jpg").write_bytes(jpgs[0])
            for k in range(1,frames_per_video):
                (out_dir/f"{stem}_w{k}.jpg").write_bytes(jpgs[k])
            strength_max = float(meta.get("strength_max", 1.0))
            strengths = [
                (i / (frames_per_video - 1)) * strength_max
                for i in range(frames_per_video)
            ]
            label = {
                "effect": effect,
                "stem": stem,
                "sample_id": sample_id,
                "tar_idx": tar_idx,
                "frame_indices": indices,
                "n_frames_total": n_frames,
                "strengths": strengths,
                "strength_max": strength_max,
                "labeling_assumption": "linear_ramp_frame0_clear",
                # Save a subset of the original metadata for reference
                "meta_subset": {
                    k: meta.get(k) for k in [
                        "effect", "strength_max", "sterngth_levels",
                        "guidance_scale", "cross_replace_steps", "self_replace_steps",
                    ]
                },
            }
            label_path.write_text(json.dumps(label, indent=2))
            n_processed += 1
            if n_processed % 50 == 0:
                print(f"  ... {n_processed} processed (+ {n_skipped} skipped)")
    
    return n_processed

def main():
    # Define what command-line arguments this script accepts.
    # If you run: python download_and_extract.py --effects fog --tars-per-effect 2
    # then args.effects == ["fog"] and args.tars_per_effect == 2.
    ap = argparse.ArgumentParser(description="Download and extract WeatherWeaver frames.")
    ap.add_argument("--effects", nargs="+", default=ALL_EFFECTS,
                    choices=ALL_EFFECTS,
                    help="Which weather effects to process (default: all six)")
    ap.add_argument("--tars-per-effect", type=int, default=1,
                    help="How many tars to download per effect (default: 1)")
    ap.add_argument("--frames-per-video", type=int, default=5,
                    help="Frames to extract per video (default: 5)")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT,
                    help=f"Output directory (default: {DEFAULT_OUT})")
    ap.add_argument("--keep-tar", action="store_true",
                    help="Keep .tar files after extraction (default: delete to save space)")
    args = ap.parse_args()
    
    # Make sure ffmpeg/ffprobe are installed before we start
    if not have_ffmpeg():
        print("ERROR: ffmpeg/ffprobe not found in PATH.", file=sys.stderr)
        print("Install with: brew install ffmpeg", file=sys.stderr)
        sys.exit(1)
    
    # Make sure the output directory exists
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    # Print a summary of what we're about to do
    print(f"Output directory: {args.out_dir}")
    print(f"Effects:          {args.effects}")
    print(f"Tars per effect:  {args.tars_per_effect}")
    print(f"Frames per video: {args.frames_per_video}")
    print(f"Keep tars:        {args.keep_tar}\n")
    
    total_processed = 0
    
    # Outer loop: each effect (cloud, fog, rain, ...)
    for effect in args.effects:
        effect_dir = args.out_dir / effect
        effect_dir.mkdir(exist_ok=True)
        
        # Inner loop: each tar within that effect (000.tar, 001.tar, ...)
        for tar_idx in range(args.tars_per_effect):
            tar_name = f"{effect}/{tar_idx:03d}.tar"
            print(f"\n{'=' * 60}")
            print(f"  {tar_name}")
            print(f"{'=' * 60}")
            
            # STEP 1: Download the tar from Hugging Face.
            # hf_hub_download caches downloads automatically, so re-runs are fast.
            print("Downloading...")
            tar_path = Path(hf_hub_download(
                repo_id="chih-hao-lin/WeatherWeaver_generation",
                filename=tar_name,
                repo_type="dataset",
            ))
            
            # STEP 2: Extract frames from every video in the tar
            print("Extracting frames...")
            n = process_one_tar(
                tar_path=tar_path,
                effect=effect,
                tar_idx=tar_idx,
                out_dir=effect_dir,
                frames_per_video=args.frames_per_video,
            )
            total_processed += n
            print(f"  Processed {n} new samples")
            
            # STEP 3: Optionally delete the tar to free up disk space.
            # The HF cache stores files as a symlink pointing to a real "blob" file.
            # We delete both the blob (the actual data) and the symlink.
            if not args.keep_tar:
                blob = tar_path.resolve()  # follows symlink to the real file
                size_mb = blob.stat().st_size / 1e6
                blob.unlink()
                tar_path.unlink(missing_ok=True)
                print(f"  Deleted tar (~{size_mb:.0f} MB)")
    
    # Final summary
    print(f"\n{'=' * 60}")
    print(f"DONE. {total_processed} new samples extracted this run.")
    print(f"Output: {args.out_dir}")
    print(f"{'=' * 60}")


# This is Python idiom: only run main() if the script is executed directly,
# not if it's imported as a module from another script.
if __name__ == "__main__":
    main()