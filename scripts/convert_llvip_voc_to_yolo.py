#!/usr/bin/env python3
"""Преобразование XML-аннотаций LLVIP из VOC в YOLO (только ИК-данные)."""

from __future__ import annotations

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


CLASS_TO_ID = {"person": 0}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--raw-dir",
        type=Path,
        default=root / "raw" / "LLVIP",
        help="Корень распакованного LLVIP с каталогами Annotations/ и infrared/",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=root / "datasets" / "person_ir",
        help="Корневой каталог выходного датасета YOLO",
    )
    p.add_argument(
        "--val-ratio",
        type=float,
        default=0.12,
        help="Доля официальной обучающей выборки, выделяемая для валидации",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--link",
        action="store_true",
        help="Создавать жёсткие/символические ссылки вместо копий (экономит место)",
    )
    return p.parse_args()


def voc_to_yolo_line(obj: ET.Element, img_w: int, img_h: int) -> str | None:
    name = (obj.findtext("name") or "").strip().lower()
    if name not in CLASS_TO_ID:
        return None
    box = obj.find("bndbox")
    if box is None:
        return None
    xmin = float(box.findtext("xmin", "0"))
    ymin = float(box.findtext("ymin", "0"))
    xmax = float(box.findtext("xmax", "0"))
    ymax = float(box.findtext("ymax", "0"))

    xmin = max(0.0, min(xmin, img_w - 1))
    ymin = max(0.0, min(ymin, img_h - 1))
    xmax = max(0.0, min(xmax, img_w - 1))
    ymax = max(0.0, min(ymax, img_h - 1))
    if xmax <= xmin or ymax <= ymin:
        return None

    xc = ((xmin + xmax) / 2.0) / img_w
    yc = ((ymin + ymax) / 2.0) / img_h
    bw = (xmax - xmin) / img_w
    bh = (ymax - ymin) / img_h
    return f"{CLASS_TO_ID[name]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


def convert_xml(xml_path: Path) -> tuple[list[str], tuple[int, int]]:
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    if size is None:
        raise ValueError(f"В файле {xml_path} отсутствует элемент <size>")
    w = int(size.findtext("width", "0"))
    h = int(size.findtext("height", "0"))
    if w <= 0 or h <= 0:
        raise ValueError(f"Некорректный размер изображения в {xml_path}: {w}x{h}")

    lines: list[str] = []
    for obj in root.findall("object"):
        line = voc_to_yolo_line(obj, w, h)
        if line:
            lines.append(line)
    return lines, (w, h)


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


def process_split(
    names: list[str],
    split: str,
    raw_dir: Path,
    out_dir: Path,
    link: bool,
) -> dict[str, int]:
    ann_dir = raw_dir / "Annotations"
    # Официальные тестовые кадры LLVIP находятся в infrared/test,
    # а обучающие — в infrared/train. Наша внутренняя валидационная выборка
    # выделяется из train, поэтому сначала ищем изображение именно там.
    img_train = raw_dir / "infrared" / "train"
    img_test = raw_dir / "infrared" / "test"

    stats = {"images": 0, "labels": 0, "boxes": 0, "missing_img": 0, "missing_xml": 0, "empty": 0}

    for name in names:
        stem = Path(name).stem
        xml_path = ann_dir / f"{stem}.xml"
        img_path = img_train / f"{stem}.jpg"
        if not img_path.exists():
            img_path = img_test / f"{stem}.jpg"

        if not xml_path.exists():
            stats["missing_xml"] += 1
            continue
        if not img_path.exists():
            stats["missing_img"] += 1
            continue

        lines, _ = convert_xml(xml_path)
        out_img = out_dir / "images" / split / f"{stem}.jpg"
        out_lbl = out_dir / "labels" / split / f"{stem}.txt"
        out_lbl.parent.mkdir(parents=True, exist_ok=True)

        place_image(img_path, out_img, link=link)
        out_lbl.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        stats["images"] += 1
        stats["labels"] += 1
        stats["boxes"] += len(lines)
        if not lines:
            stats["empty"] += 1

    return stats


def write_data_yaml(out_dir: Path) -> None:
    yaml_path = out_dir / "data.yaml"
    yaml_path.write_text(
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
    raw_dir = args.raw_dir
    out_dir = args.out_dir

    train_dir = raw_dir / "infrared" / "train"
    if not train_dir.is_dir():
        raise SystemExit(f"Не найден каталог {train_dir}. Сначала распакуйте LLVIP infrared.")

    train_names = sorted(p.name for p in train_dir.glob("*.jpg"))
    if not train_names:
        raise SystemExit(f"В каталоге {train_dir} нет изображений")

    rng = random.Random(args.seed)
    shuffled = train_names[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * args.val_ratio))
    val_names = sorted(shuffled[:n_val])
    train_split = sorted(shuffled[n_val:])

    print(f"[LLVIP] всего обучающих изображений: {len(train_names)}")
    print(
        f"[LLVIP] -> YOLO train: {len(train_split)}, val: {len(val_names)} "
        f"(доля val={args.val_ratio})"
    )

    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    train_stats = process_split(train_split, "train", raw_dir, out_dir, args.link)
    val_stats = process_split(val_names, "val", raw_dir, out_dir, args.link)
    write_data_yaml(out_dir)

    print("[OK] преобразование завершено")
    print(f"  train: {train_stats}")
    print(f"  val:   {val_stats}")
    print(f"  data:  {out_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()
