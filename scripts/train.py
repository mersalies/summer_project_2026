#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


# ======== настройки ==============

# Датасет
# DATASET = "llvip"   # LLVIP IR
DATASET = "flir"      # FLIR thermal (person)

# Стартовые веса (при MODE=resume не используется)
# START = "coco"      # yolov8n.pt
START = "llvip"       # от best LLVIP
# START = "flir"      # от best FLIR

# Режим
# MODE = "smoke"      # 3 эпохи
MODE = "full"         # до 80 эпох
# MODE = "resume"     # продолжить last.pt этого DATASET
# MODE = "low_vram"   # batch=4, imgsz=512

# остановка если val не растёт 15 эпох
EARLY_STOP_PATIENCE = 15

# =============================================================================

CFG = {
    "llvip": {
        "data": ROOT / "datasets" / "person_ir" / "data.yaml",
        "name": "llvip_ir_yolov8n",
        "best_out": ROOT / "weights" / "best_llvip_ir_yolov8n.pt",
        "hint": "python scripts/convert_llvip_voc_to_yolo.py --link",
    },
    "flir": {
        "data": ROOT / "datasets" / "person_flir" / "data.yaml",
        "name": "flir_thermal_yolov8n",
        "best_out": ROOT / "weights" / "best_flir_thermal_yolov8n.pt",
        "hint": "python scripts/convert_flir_coco_person_to_yolo.py --link",
    },
}

START_PT = {
    "coco": "yolov8n.pt",
    "llvip": ROOT / "weights" / "best_llvip_ir_yolov8n.pt",
    "flir": ROOT / "weights" / "best_flir_thermal_yolov8n.pt",
}


def apply_button_settings(args: argparse.Namespace) -> None:
    """Применить DATASET / START / MODE / EARLY_STOP"""
    mode = MODE.strip().lower()
    if mode == "smoke":
        args.smoke, args.resume = True, False
    elif mode == "full":
        args.smoke, args.resume, args.epochs = False, False, 80
    elif mode == "resume":
        args.smoke, args.resume = False, True
    elif mode == "low_vram":
        args.smoke, args.resume, args.epochs = False, False, 80
        args.batch, args.imgsz, args.workers = 4, 512, 2
    else:
        raise SystemExit(f"[ОШИБКА] MODE='{mode}'")

    ds = DATASET.strip().lower()
    if ds not in CFG:
        raise SystemExit(f"[ОШИБКА] DATASET='{ds}'. Допустимо: llvip, flir")
    cfg = CFG[ds]
    args.data, args.name = cfg["data"], cfg["name"]
    args._best_out, args._hint = cfg["best_out"], cfg["hint"]

    patience = int(EARLY_STOP_PATIENCE)
    args.patience = max(args.epochs * 2, 999) if patience == 0 else patience

    print(f"[MODE] {mode}")
    print(f"[DATASET] {ds} → {args.data}")
    print(f"[EARLY STOP] patience={args.patience}")

    if args.resume:
        print("[START] resume → last.pt")
        return

    start = START.strip().lower()
    if start not in START_PT:
        raise SystemExit(f"[ОШИБКА] START='{start}'. Допустимо: coco, llvip, flir")
    w = START_PT[start]
    if start != "coco" and not Path(w).exists():
        raise SystemExit(f"[ОШИБКА] нет весов: {w}")
    args.model = str(w)
    print(f"[START] {start} → {args.model}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Обучение YOLOv8n на IR/thermal.")
    p.add_argument("--data", type=Path, default=CFG["llvip"]["data"])
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--project", type=Path, default=ROOT / "runs" / "detect")
    p.add_argument("--name", default="llvip_ir_yolov8n")
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--save-period", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr0", type=float, default=0.01)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--cache", default="False", choices=["True", "False", "ram", "disk"])
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args._best_out = CFG["llvip"]["best_out"]
    args._hint = CFG["llvip"]["hint"]

    if len(sys.argv) == 1:
        apply_button_settings(args)

    if not args.data.exists():
        print(f"[ОШИБКА] нет {args.data}\nЗапустите: {args._hint}", file=sys.stderr)
        return 1

    try:
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        print(f"[ОШИБКА] {exc}\n  source .venv/bin/activate && uv pip install -r requirements.txt")
        return 1

    epochs = 3 if args.smoke else args.epochs
    cache = False if args.cache == "False" else True if args.cache == "True" else args.cache

    print("=" * 60)
    print("Обучение YOLO (человек, тепловой спектр)")
    for k, v in [
        ("data", args.data), ("model", args.model),
        ("epochs", f"{epochs}{' (smoke)' if args.smoke else ''}"),
        ("imgsz", args.imgsz), ("batch", args.batch), ("device", args.device),
        ("patience", args.patience), ("resume", args.resume),
        ("run", args.project / args.name),
    ]:
        print(f"  {k:10}: {v}")
    if torch.cuda.is_available() and args.device != "cpu":
        props = torch.cuda.get_device_properties(int(args.device) if str(args.device).isdigit() else 0)
        print(f"  {'GPU':10}: {props.name} ({props.total_memory / 1024**3:.1f} GiB)")
    print("=" * 60)

    model_path = args.model
    if args.resume:
        last = args.project / args.name / "weights" / "last.pt"
        if not last.exists():
            print(f"[ОШИБКА] нет checkpoint: {last}", file=sys.stderr)
            return 1
        model_path = str(last)
        print(f"[RESUME] {last}")

    model = YOLO(str(model_path))
    try:
        results = model.train(
            data=str(args.data),
            epochs=epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            project=str(args.project),
            name=args.name,
            exist_ok=True,
            patience=args.patience,
            seed=args.seed,
            lr0=args.lr0,
            amp=args.amp,
            resume=args.resume,
            cache=cache,
            save_period=args.save_period,
            hsv_h=0.005,
            hsv_s=0.3,
            hsv_v=0.3,
            fliplr=0.5,
            mosaic=0.5,
            close_mosaic=10,
            plots=True,
            save=True,
            verbose=True,
        )
    except torch.cuda.OutOfMemoryError:
        print("[OOM] поставьте MODE = \"low_vram\" и запустите снова.", file=sys.stderr)
        return 2

    save_dir = Path(getattr(results, "save_dir", args.project / args.name))
    best = save_dir / "weights" / "best.pt"
    if best.exists():
        out = Path(args._best_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(best.read_bytes())
        print(f"[OK] лучшие веса -> {out}")
    print(f"[OK] прогон: {save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
