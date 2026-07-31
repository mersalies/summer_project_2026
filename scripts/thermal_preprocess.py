#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError as exc: 
    raise SystemExit("Нужен opencv-python: " + str(exc))


def preprocess_frame(frame: np.ndarray, mode: str = "none") -> np.ndarray:
    """none | whitehot/gray | invert → 3-канальный BGR для YOLO."""
    if mode == "none":
        return frame
    if mode in ("whitehot", "gray"):
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif mode == "invert":
        g = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(g)
    else:
        raise ValueError(f"Режим '{mode}'. Допустимо: none, whitehot, gray, invert")
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
