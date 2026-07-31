#!/usr/bin/env python3
"""Конвертация FLIR ADAS v2 thermal COCO → YOLO (только класс person)."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--raw-root",
        type=Path,
        default=root / "raw" / "FLIR_ADAS_v2",
        help="Корень распакованного FLIR ADAS v2",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=root / "datasets" / "person_flir",
        help="Выходной YOLO-датасет",
    )
    p.add_argument(
        "--link",
        action="store_true",
        help="Жёсткие/символические ссылки вместо копирования изображений",
    )
    p.add_argument(
        "--keep-empty",
        action="store_true",
        help="Оставлять кадры без person (как негативы). По умолчанию только кадры с людьми.",
    )
    return p.parse_args()


def place_image(src: Path, dst: Path, link: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if link:
        try:
            dst.hardlink_to(src)
        except OSError:
            try:
                dst.symlink_to(src.resolve())
            except OSError:
                shutil.copy2(src, dst)
    else:
        shutil.copy2(src, dst)


def coco_bbox_to_yolo(bbox: list[float], img_w: int, img_h: int) -> str | None:
    """COCO [x, y, w, h] в пикселях → YOLO class xc yc bw bh (норм.)."""
    x, y, w, h = map(float, bbox)
    if w <= 0 or h <= 0 or img_w <= 0 or img_h <= 0:
        return None
    xc = (x + w / 2.0) / img_w
    yc = (y + h / 2.0) / img_h
    bw = w / img_w
    bh = h / img_h
    # Отсечь вырожденные / сильно вылезшие за кадр боксы.
    if bw <= 0 or bh <= 0:
        return None
    xc = min(max(xc, 0.0), 1.0)
    yc = min(max(yc, 0.0), 1.0)
    bw = min(max(bw, 0.0), 1.0)
    bh = min(max(bh, 0.0), 1.0)
    return f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


def convert_split(
    coco_path: Path,
    images_dir: Path,
    out_dir: Path,
    split: str,
    link: bool,
    keep_empty: bool,
) -> dict[str, int]:
    data = json.loads(coco_path.read_text(encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in data["categories"]}
    person_ids = {cid for cid, name in cats.items() if name.lower() == "person"}
    if not person_ids:
        raise SystemExit(f"В {coco_path} нет категории person")

    images = {im["id"]: im for im in data["images"]}
    by_image: dict[int, list[str]] = {iid: [] for iid in images}

    for ann in data["annotations"]:
        if ann.get("category_id") not in person_ids:
            continue
        im = images.get(ann["image_id"])
        if im is None:
            continue
        line = coco_bbox_to_yolo(ann["bbox"], int(im["width"]), int(im["height"]))
        if line:
            by_image[ann["image_id"]].append(line)

    stats = {
        "images": 0,
        "labels": 0,
        "boxes": 0,
        "skipped_no_person": 0,
        "missing_img": 0,
    }

    for iid, im in images.items():
        lines = by_image.get(iid, [])
        if not lines and not keep_empty:
            stats["skipped_no_person"] += 1
            continue

        file_name = im["file_name"]
        # file_name обычно вида data/video-....jpg
        src = images_dir / file_name
        if not src.exists():
            src = images_dir / "data" / Path(file_name).name
        if not src.exists():
            stats["missing_img"] += 1
            continue

        # Уникальное имя без коллизий между роликами.
        stem = Path(file_name).stem
        out_img = out_dir / "images" / split / f"{stem}.jpg"
        out_lbl = out_dir / "labels" / split / f"{stem}.txt"
        out_lbl.parent.mkdir(parents=True, exist_ok=True)

        place_image(src, out_img, link=link)
        out_lbl.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        stats["images"] += 1
        stats["labels"] += 1
        stats["boxes"] += len(lines)

    return stats


def write_data_yaml(out_dir: Path) -> None:
    (out_dir / "data.yaml").write_text(
        "\n".join(
            [
                f"path: {out_dir.resolve()}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: person",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    raw = args.raw_root
    out = args.out_dir

    train_coco = raw / "images_thermal_train" / "coco.json"
    val_coco = raw / "images_thermal_val" / "coco.json"
    if not train_coco.exists() or not val_coco.exists():
        raise SystemExit(
            f"Не найдены coco.json. Сначала распакуйте thermal:\n"
            f"  bash scripts/unpack_flir_thermal.sh"
        )

    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    print("[FLIR] конвертация thermal → YOLO (только person)...")
    train_stats = convert_split(
        train_coco,
        raw / "images_thermal_train",
        out,
        "train",
        args.link,
        args.keep_empty,
    )
    val_stats = convert_split(
        val_coco,
        raw / "images_thermal_val",
        out,
        "val",
        args.link,
        args.keep_empty,
    )
    write_data_yaml(out)

    print("[OK] преобразование завершено")
    print(f"  train: {train_stats}")
    print(f"  val:   {val_stats}")
    print(f"  data:  {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
