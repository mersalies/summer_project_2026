#!/usr/bin/env bash
# Распаковка только тепловой части FLIR ADAS v2 без RGB-данных.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ZIP="$ROOT/archive.zip"
OUT="$ROOT/raw"

if [[ ! -f "$ZIP" ]]; then
  echo "[ОШИБКА] Не найден архив $ZIP"
  exit 1
fi

mkdir -p "$OUT"
echo "[FLIR] Распаковка thermal train/val и video_thermal_test (без RGB)..."
unzip -n -q "$ZIP" \
  "FLIR_ADAS_v2/images_thermal_train/*" \
  "FLIR_ADAS_v2/images_thermal_val/*" \
  "FLIR_ADAS_v2/video_thermal_test/*" \
  "README.txt" \
  "rgb_to_thermal_vid_map.json" \
  -d "$OUT"

echo "[OK] Тепловая часть FLIR распакована в $OUT/FLIR_ADAS_v2"
du -sh "$OUT/FLIR_ADAS_v2" 2>/dev/null || true
find "$OUT/FLIR_ADAS_v2" -maxdepth 2 -type d | sort
