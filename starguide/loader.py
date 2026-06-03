"""Robust image/source loading + source-type detection.

`imread` reads ordinary formats with OpenCV and falls back to ffmpeg for exotic
ones (AVIF/HEIC/WebP) so any photo a user throws at us decodes. `is_video`
decides one-off-image vs live-stream/clip so the unified entry can route to the
blind solver or the motion solver.
"""

from __future__ import annotations

import os
import subprocess
import tempfile

import cv2
import numpy as np

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".mpg", ".mpeg"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif",
              ".avif", ".heic", ".heif", ".webp"}


def imread(path: str) -> np.ndarray:
    """Read an image as BGR, transcoding via ffmpeg if OpenCV can't.

    Reads the bytes ourselves so non-ASCII paths work (cv2.imread silently fails
    on the narrow-no-break-space macOS uses in screenshot names)."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size:
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is not None:
                return img
    except (OSError, ValueError):
        pass
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        out = tmp.name
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path, out],
                       check=True)
        img = cv2.imread(out)
    finally:
        if os.path.exists(out):
            os.remove(out)
    if img is None:
        raise ValueError(f"could not decode image {path!r} "
                         "(install ffmpeg for AVIF/HEIC support)")
    return img


def is_video(path: str) -> bool:
    """True for a multi-frame source (clip/stream), False for a still image."""
    ext = os.path.splitext(path)[1].lower()
    if ext in VIDEO_EXTS:
        return True
    if ext in IMAGE_EXTS:
        return False
    cap = cv2.VideoCapture(path)          # unknown extension: probe frame count
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n > 1
