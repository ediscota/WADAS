#!/usr/bin/env python3
# Description: Extract evenly spaced frames from the test videos as JPEG photos, the input of
# the camera load test.

"""Create the photo set used by run_load.py by extracting frames from the test videos."""

import argparse
from pathlib import Path

import cv2


def extract_frames(video, out_dir, per_video, quality=90):
    """Save per_video evenly spaced frames of video as JPEG files; return how many were saved."""
    capture = cv2.VideoCapture(str(video))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    saved = 0
    for index in range(per_video):
        position = int(total * (index + 0.5) / per_video)
        capture.set(cv2.CAP_PROP_POS_FRAMES, position)
        ok, frame = capture.read()
        if not ok:
            continue
        target = out_dir / f"{video.stem}_f{position:05d}.jpg"
        cv2.imwrite(str(target), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        saved += 1
    capture.release()
    return saved


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos-dir", type=Path, default=Path.home() / "VIDEO_TEST")
    parser.add_argument("--out-dir", type=Path, default=Path.home() / "PHOTO_TEST")
    parser.add_argument("--per-video", type=int, default=4)
    args = parser.parse_args(argv)

    videos = sorted(args.videos_dir.expanduser().glob("*.mp4"))
    if not videos:
        parser.error(f"no .mp4 files in {args.videos_dir}")
    out_dir = args.out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    total = sum(extract_frames(video, out_dir, args.per_video) for video in videos)
    print(f"{total} photos from {len(videos)} videos in {out_dir}")


if __name__ == "__main__":
    main()
