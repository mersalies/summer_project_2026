#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FRAMES = ROOT / "raw" / "FLIR_ADAS_v2" / "video_thermal_test" / "data"
FRAME_RE = re.compile(r"(video-[A-Za-z0-9]+)-frame-(\d+)-")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frames-dir", type=Path, default=DEFAULT_FRAMES)
    p.add_argument(
        "--video-id",
        default=None,
        help="Идентификатор video-XXXX. По умолчанию выбирается ролик с максимумом кадров.",
    )
    p.add_argument("--fps", type=int, default=30)
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "demos" / "flir_thermal_clip.mp4",
    )
    p.add_argument("--limit", type=int, default=0, help="Максимум кадров (0 = все)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    frames_dir = args.frames_dir
    if not frames_dir.is_dir():
        raise SystemExit(f"Каталог с кадрами не найден: {frames_dir}")

    groups: dict[str, list[tuple[int, Path]]] = {}
    for fp in frames_dir.glob("*.jpg"):
        m = FRAME_RE.search(fp.name)
        if not m:
            continue
        groups.setdefault(m.group(1), []).append((int(m.group(2)), fp))

    if not groups:
        raise SystemExit("Не найдены кадры FLIR с ожидаемой схемой имён.")

    if args.video_id:
        vid = args.video_id if args.video_id.startswith("video-") else f"video-{args.video_id}"
        if vid not in groups:
            raise SystemExit(f"Неизвестный ID видео {vid}. Доступны: {sorted(groups)}")
    else:
        vid = max(groups, key=lambda k: len(groups[k]))

    frames = sorted(groups[vid], key=lambda t: t[0])
    if args.limit > 0:
        frames = frames[: args.limit]
    print(f"[FLIR] видео {vid}: {len(frames)} кадров, {args.fps} кадров/с")

    first = cv2.imread(str(frames[0][1]))
    if first is None:
        raise SystemExit(f"Не удалось прочитать {frames[0][1]}")
    h, w = first.shape[:2]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.out), fourcc, args.fps, (w, h))
    for i, (_, fp) in enumerate(frames):
        img = cv2.imread(str(fp))
        if img is None:
            continue
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        writer.write(img)
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(frames)}")
    writer.release()
    print(f"[OK] видео записано: {args.out} ({w}x{h})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
