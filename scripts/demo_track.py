#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


# =================== НАСТРОЙКИ ======================== 

WEIGHTS = "flir"  # "flir" | "llvip"
SOURCE = "demos/eval_holdout_1.mp4"
PREPROCESS = "none"  # "none" | "whitehot" | "invert"
CONF = 0.25
SHOW = False  # True = онлайн окно
SAVE = True
# =============================================================================

WEIGHT_FILES = {
    "flir": ROOT / "weights" / "best_flir_thermal_yolov8n.pt",
    "llvip": ROOT / "weights" / "best_llvip_ir_yolov8n.pt",
}


def ensure_gui() -> None:
    import os
    import cv2

    if getattr(getattr(cv2, "version", None), "headless", False) or not hasattr(cv2, "imshow"):
        raise SystemExit("[ОШИБКА] нужен opencv-python (не headless)")
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise SystemExit("[ОШИБКА] нет DISPLAY. Запускайте из Konsole, не из VS Code Terminal.")
    print("[SHOW] окно откроется; q = выход. На Wayland: QT_QPA_PLATFORM=xcb")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Детекция + трекинг на IR/thermal.")
    p.add_argument("--weights", type=Path, default=None)
    p.add_argument("--source", default=None)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=None)
    p.add_argument("--device", default="0")
    p.add_argument("--tracker", default="bytetrack.yaml")
    p.add_argument("--project", type=Path, default=ROOT / "demos")
    p.add_argument("--name", default="track")
    p.add_argument("--save", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--show", action="store_true")
    p.add_argument("--preprocess", choices=["none", "whitehot", "gray", "invert"], default=None)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--max-frames", type=int, default=0)
    return p.parse_args()


def apply_button_defaults(args: argparse.Namespace) -> None:
    if args.weights is None:
        key = WEIGHTS.strip().lower()
        if key not in WEIGHT_FILES:
            raise SystemExit(f"[ОШИБКА] WEIGHTS='{key}'. Допустимо: flir, llvip")
        args.weights = WEIGHT_FILES[key]
    if args.source is None:
        args.source = SOURCE
    if args.conf is None:
        args.conf = CONF
    if args.preprocess is None:
        args.preprocess = PREPROCESS
    if args.save is None:
        args.save = SAVE
    if not args.show:
        args.show = SHOW


def iter_frames(source: str):
    import cv2

    path = Path(source)
    if path.is_dir():
        for fp in sorted(path.glob("*")):
            if fp.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                img = cv2.imread(str(fp))
                if img is not None:
                    yield img
        return
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else str(source))
    if not cap.isOpened():
        raise SystemExit(f"Не открыть источник: {source}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


def run_plain(args) -> Path:
    from ultralytics import YOLO

    if args.show:
        ensure_gui()
    YOLO(str(args.weights)).track(
        source=str(args.source),
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        tracker=args.tracker,
        persist=True,
        save=args.save,
        show=args.show,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        verbose=True,
    )
    return args.project / args.name


def run_preprocess(args) -> Path:
    import cv2
    from ultralytics import YOLO
    from thermal_preprocess import preprocess_frame

    if args.show:
        ensure_gui()

    out_dir = args.project / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_mp4 = out_dir / "demo.mp4"
    model = YOLO(str(args.weights))
    writer = None
    n = 0
    print(f"[PREP] {args.preprocess}")

    for frame in iter_frames(str(args.source)):
        prepped = preprocess_frame(frame, mode=args.preprocess)
        res = model.track(
            prepped,
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            tracker=args.tracker,
            persist=True,
            verbose=False,
        )
        annotated = res[0].plot()
        if args.save:
            if writer is None:
                h, w = annotated.shape[:2]
                writer = cv2.VideoWriter(
                    str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h)
                )
            writer.write(annotated)
        if args.show:
            cv2.imshow("YOLO thermal (q=выход)", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        n += 1
        if args.max_frames and n >= args.max_frames:
            break

    if writer:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()
    print(f"[OK] кадров: {n}" + (f", видео: {out_mp4}" if args.save else ""))
    return out_dir


def avi_to_mp4(out_dir: Path) -> None:
    avis = sorted(out_dir.glob("*.avi"))
    if not avis:
        return
    mp4 = out_dir / "demo.mp4"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(avis[0]),
             "-c:v", "libx264", "-pix_fmt", "yuv420p", str(mp4)],
            check=True,
        )
        print(f"[OK] откройте: {mp4}")
    except Exception as exc:
        print(f"[WARN] mp4 не создан ({exc}). Откройте: {avis[0]}")


def main() -> int:
    args = parse_args()
    if len(sys.argv) == 1:
        apply_button_defaults(args)
    else:
        # CLI без button-блока: заполнить пробелы разумными дефолтами
        if args.weights is None:
            args.weights = WEIGHT_FILES["flir"]
        if args.source is None:
            print("[ОШИБКА] укажите --source", file=sys.stderr)
            return 1
        if args.conf is None:
            args.conf = 0.25
        if args.preprocess is None:
            args.preprocess = "none"
        if args.save is None:
            args.save = True

    if not args.weights.exists():
        print(f"[ОШИБКА] нет весов: {args.weights}", file=sys.stderr)
        return 1
    if not str(args.source).isdigit() and not Path(args.source).exists():
        print(f"[ОШИБКА] нет источника: {args.source}", file=sys.stderr)
        return 1

    print("=" * 50)
    print(f"  weights : {args.weights}")
    print(f"  source  : {args.source}")
    print(f"  prep    : {args.preprocess}")
    print(f"  conf    : {args.conf}")
    print("=" * 50)

    if args.preprocess == "none":
        out = run_plain(args)
        print(f"[OK] {out}")
        if args.save:
            avi_to_mp4(out)
    else:
        run_preprocess(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
