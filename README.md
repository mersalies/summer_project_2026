# Детекция и трекинг людей в тепловом спектре

Документация проекта летней практики.  
Цель: научить нейросеть **находить людей** на тепловых (ИК) кадрах и **сопровождать** их номерами (track ID) на видео.

---

## 1. О чём проект простыми словами

Есть тепловизор (или тепловое видео). Люди на нём выглядят как светлые/тёмные силуэты.

Мы берём готовую нейросеть **YOLOv8n**, дообучаем её на тепловых датасетах (сначала LLVIP, потом FLIR) и получаем модель, которая:

1. рисует рамку вокруг человека;
2. пишет класс `person` и уверенность;
3. на видео присваивает человеку номер (ID) и старается держать его между кадрами.

---

## 2. Какая модель используется

| Параметр | Значение |
|---|---|
| Архитектура | **YOLOv8n** (nano) — самая лёгкая YOLOv8 |
| Задача | `detect` (детекция боксами) |
| Класс | один: `person` |
| Библиотека | Ultralytics |
| Трекер | ByteTrack (внутри `model.track`) |
| Железо при обучении | RTX 3050 Laptop 4 ГБ, batch=8, imgsz=640, AMP |

### Готовые веса в проекте

| Файл | Что это |
|---|---|
| `weights/best_llvip_ir_yolov8n.pt` | после обучения на LLVIP (~40 эпох) |
| `weights/best_flir_thermal_yolov8n.pt` | **актуальная модель**: LLVIP → дообучение на FLIR |

Для демо и камеры по умолчанию берите **`best_flir_thermal_yolov8n.pt`**.

### Палитра изображений

Модель училась на **white-hot**: тёплое = светлое, холодное = тёмное.  
Если камера отдаёт цветной «ironbow» — перед моделью кадр переводится в white-hot (`PREPROCESS = "whitehot"`).

---

## 3. Архитектура проекта

### 3.1. Как устроен поток данных

```text
Архивы датасетов (LLVIP.zip, archive.zip)
        │
        ▼
raw/                          ← распакованные «как есть»
        │  скрипты convert_*.py
        ▼
datasets/person_ir            ← LLVIP в формате YOLO
datasets/person_flir          ← FLIR (только person) в формате YOLO
        │  scripts/train.py
        ▼
runs/detect/.../weights/best.pt
        │  копирование
        ▼
weights/best_*.pt             ← удобные веса для демо
        │  scripts/demo_track.py
        ▼
видео / камера
        │
        ▼
demos/.../demo.mp4            ← видео с рамками и ID
```

### 3.2. Структура папок

```text
Summer_project_2026/
│
├── README.md                 ← этот документ
├── requirements.txt          ← зависимости Python
├── .venv/                    ← виртуальное окружение (Python 3.12)
├── yolov8n.pt                ← стартовые веса COCO (скачиваются Ultralytics)
│
├── LLVIP.zip                 ← исходный архив LLVIP
├── archive.zip               ← исходный архив FLIR ADAS v2
│
├── raw/                      ← распакованные данные (не трогать руками)
│   ├── LLVIP/                ← Annotations + infrared
│   └── FLIR_ADAS_v2/         ← thermal train/val + video_thermal_test
│
├── datasets/                 ← данные уже в формате YOLO
│   ├── person_ir/            ← LLVIP IR
│   │   ├── data.yaml
│   │   ├── images/{train,val}/
│   │   └── labels/{train,val}/
│   └── person_flir/          ← FLIR thermal, только класс person
│       ├── data.yaml
│       ├── images/{train,val}/
│       └── labels/{train,val}/
│
├── scripts/                  ← весь код проекта
│   ├── train.py              ← обучение
│   ├── demo_track.py         ← прогон видео/камеры
│   ├── thermal_preprocess.py ← перевод палитры
│   ├── convert_llvip_voc_to_yolo.py
│   ├── convert_flir_coco_person_to_yolo.py
│   ├── build_flir_video.py
│   └── unpack_flir_thermal.sh
│
├── weights/                  ← лучшие модели для запуска
│   ├── best_llvip_ir_yolov8n.pt
│   └── best_flir_thermal_yolov8n.pt
│
├── runs/detect/              ← логи и графики обучения
│   ├── llvip_ir_yolov8n/
│   └── flir_thermal_yolov8n/
│
└── demos/                    ← входные/выходные видео
    ├── eval_holdout_1.mp4    ← тестовое видео (не участвовало в обучении)
    ├── eval_holdout_2.mp4
    ├── eval_flir_holdout_1/demo.mp4  ← результат с рамками
    └── eval_flir_holdout_2/demo.mp4
```

### 3.3. Почему папки разделены

| Папка | Смысл |
|---|---|
| `raw/` | оригиналы датасета, чтобы всегда можно было пересобрать |
| `datasets/` | то, что «ест» YOLO (картинка + `.txt` с рамками) |
| `weights/` | короткие пути к лучшим моделям |
| `runs/` | полная история эксперимента (графики, args, best/last) |
| `demos/` | всё, что связано с просмотром результата |

---

## 4. Как пользоваться «для чайника»

Ниже — путь от нуля до видео с рамками.  
Если окружение и датасеты уже подготовлены (как сейчас), можно сразу переходить к шагам 4–6.

### Шаг 0. Открыть проект

1. Откройте папку проекта в VS Code / Cursor.
2. Внизу справа (статусбар) должен быть интерпретатор **`.venv (Python 3.12)`**.  
   Если другой — кликните и выберите `.venv`.

Проверка в терминале:

```bash
cd "/home/arch/VS Code/Project Python/Summer_project_2026"
source .venv/bin/activate
python -c "import torch; print(torch.cuda.is_available())"
```

Должно напечатать `True` (если есть NVIDIA GPU).

---

### Шаг 1. (Опционально) Если окружения ещё нет

```bash
cd "/home/arch/VS Code/Project Python/Summer_project_2026"
uv python install 3.12
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

---

### Шаг 2. (Опционально) Подготовка данных — уже сделано

Обычно **заново делать не нужно**. Нужно только если удалили `datasets/` или `raw/`.

```bash
source .venv/bin/activate

# LLVIP IR
unzip -n -q LLVIP.zip "LLVIP/Annotations/*" "LLVIP/infrared/*" -d raw/
python scripts/convert_llvip_voc_to_yolo.py --link

# FLIR thermal (без RGB)
bash scripts/unpack_flir_thermal.sh
python scripts/convert_flir_coco_person_to_yolo.py --link
```

---

### Шаг 3. Обучить / дообучить модель

1. Откройте файл `scripts/train.py`.
2. Вверху найдите блок настроек.
3. Оставьте **одну** активную строку в каждом разделе:

```python
DATASET = "flir"      # на каком датасете учим
START = "llvip"       # от каких весов стартуем
MODE = "full"         # smoke / full / resume / low_vram
EARLY_STOP_PATIENCE = 15
```

4. Нажмите ▶ справа сверху → **Run Python File**.
5. Смотрите прогресс в терминале: эпохи, %, loss, mAP.
6. Когда закончится — лучшие веса появятся в `weights/`.

**Типичные варианты:**

| Цель | DATASET | START | MODE |
|---|---|---|---|
| Быстрая проверка | `flir` | `llvip` | `smoke` |
| Полное дообучение FLIR от LLVIP | `flir` | `llvip` | `full` |
| Продолжить прерванный FLIR | `flir` | любой | `resume` |
| Вернуться к LLVIP | `llvip` | `llvip` | `full` |
| Мало видеопамяти | любой | любой | `low_vram` |

Остановка вручную: `Ctrl+C`.  
Автостоп: если метрика на проверке не растёт `EARLY_STOP_PATIENCE` эпох.

---

### Шаг 4. Прогнать готовое видео через модель (самое нужное)

#### Способ А — кнопкой в VS Code (проще)

1. Откройте `scripts/demo_track.py`.
2. Вверху выставьте:

```python
WEIGHTS = "flir"
SOURCE = "demos/eval_holdout_1.mp4"
PREPROCESS = "none"
CONF = 0.25
SHOW = False
SAVE = True
```

3. ▶ **Run Python File**.
4. Дождитесь конца.
5. Откройте результат:

```text
demos/track/demo.mp4
```

(или папку, указанную в `--name`; по умолчанию имя прогона `track`).

#### Способ Б — командой в терминале

```bash
source .venv/bin/activate

python scripts/demo_track.py \
  --source demos/eval_holdout_1.mp4 \
  --weights weights/best_flir_thermal_yolov8n.pt \
  --conf 0.25 \
  --name my_demo
```

Результат: `demos/my_demo/demo.mp4`.

На видео вы увидите:
- синие/цветные рамки вокруг людей;
- подпись `person` и число уверенности;
- `id:N` — номер человека в трекере.

---

### Шаг 5. Смотреть детекции в реальном времени (живое окно)

Во встроенном терминале VS Code окно часто **не открывается**.  
Откройте **системный терминал** (Konsole / Alacritty):

```bash
cd "/home/arch/VS Code/Project Python/Summer_project_2026"
source .venv/bin/activate

python scripts/demo_track.py \
  --source demos/eval_holdout_1.mp4 \
  --weights weights/best_flir_thermal_yolov8n.pt \
  --conf 0.25 \
  --show \
  --no-save
```

Или в `demo_track.py` поставьте `SHOW = True` и запускайте из системного терминала.

- Клавиша **q** — выход из окна.
- Если на Wayland окно не появляется:

```bash
QT_QPA_PLATFORM=xcb python scripts/demo_track.py ... --show --no-save
```

---

### Шаг 6. Подключить камеру

#### 6.1. Веб-камера ноутбука (для проверки пайплайна)

```bash
source .venv/bin/activate

python scripts/demo_track.py \
  --source 0 \
  --weights weights/best_flir_thermal_yolov8n.pt \
  --conf 0.25 \
  --show \
  --no-save
```

`--source 0` — камера с индексом 0.  
Если камер несколько: попробуйте `1`, `2`.

**Важно:** обычная веб-камера — это **обычный цветной** кадр, а модель училась на **тепловом**.  
Качество на вебке будет плохим — это нормально. Вебка нужна только чтобы проверить, что «камера открывается → кадры идут → окно рисуется».

#### 6.2. Тепловизор / USB-тепловая камера

1. Подключите устройство.
2. Узнайте, как оно видно в системе (часто тоже `/dev/videoN` → индекс `0/1/2`).
3. Запустите как камеру:

```bash
# если тепловизор уже отдаёт white-hot (ч/б):
python scripts/demo_track.py --source 0 --preprocess none --show --no-save

# если отдаёт псевдоцвет (жёлто-фиолетовый ironbow):
python scripts/demo_track.py --source 0 --preprocess whitehot --show --no-save

# если black-hot (люди тёмные):
python scripts/demo_track.py --source 0 --preprocess invert --show --no-save
```

В блоке настроек `demo_track.py` то же самое:

```python
WEIGHTS = "flir"
SOURCE = "0"
PREPROCESS = "whitehot"   # или none / invert
SHOW = True
SAVE = False
```

#### 6.3. Если камера не открывается

- закройте Zoom/Telegram/Cheese — они могут держать камеру;
- проверьте права на `/dev/video*`;
- смените индекс: `--source 1`;
- убедитесь, что запускаете не по SSH без экрана.

---

### Шаг 7. Собрать своё тестовое видео из кадров FLIR (опционально)

Hold-out ролики уже лежат в `demos/eval_holdout_*.mp4`.  
Если нужен другой:

```bash
python scripts/build_flir_video.py --video-id ZAtDSNuZZjkZFvMAo --out demos/my_clip.mp4
python scripts/demo_track.py --source demos/my_clip.mp4 --conf 0.25 --name my_clip_out
```

Эти `video_thermal_test` **не входят в обучение** — ими можно честно оценивать модель.

---

## 5. Что делает каждый скрипт

### `scripts/train.py`

**Зачем:** обучить или дообучить YOLO.

**Как запускать:** ▶ Run Python File (или из терминала с флагами).

**Что внутри:**
- читает блок настроек `DATASET` / `START` / `MODE` / `EARLY_STOP_PATIENCE`;
- подставляет путь к `data.yaml` и стартовые `.pt`;
- вызывает `YOLO(...).train(...)`;
- сохраняет прогон в `runs/detect/<name>/`;
- копирует лучшие веса в `weights/best_*.pt`.

**Выход:**
- `runs/detect/.../weights/best.pt`, `last.pt`
- графики: `results.png`, PR-кривые, confusion matrix
- копия best → `weights/`

---

### `scripts/demo_track.py`

**Зачем:** прогнать модель по видео/папке кадров/камере и получить рамки + ID.

**Как запускать:** ▶ или терминал.

**Что внутри:**
- загружает веса;
- если `PREPROCESS != none` — каждый кадр приводит к white-hot;
- вызывает `model.track(...)` (детекция + ByteTrack);
- рисует рамки;
- сохраняет `demo.mp4` и/или показывает окно (`SHOW=True`).

**Выход:** `demos/<name>/demo.mp4`.

---

### `scripts/thermal_preprocess.py`

**Зачем:** вспомогательный модуль для `demo_track.py`.

**Режимы:**
- `none` — ничего не менять;
- `whitehot` / `gray` — сделать кадр «тёплое = светлое»;
- `invert` — инвертировать яркость (black-hot → white-hot).

Сам по себе обычно не запускается.

---

### `scripts/convert_llvip_voc_to_yolo.py`

**Зачем:** превратить разметку LLVIP из VOC XML в YOLO TXT.

**Вход:** `raw/LLVIP/Annotations` + `raw/LLVIP/infrared`  
**Выход:** `datasets/person_ir/`

Делает train/val split, пишет `data.yaml`, умеет `--link` (ссылки вместо копий).

---

### `scripts/convert_flir_coco_person_to_yolo.py`

**Зачем:** из FLIR COCO взять **только класс person** и сделать YOLO-датасет.

**Вход:** `raw/FLIR_ADAS_v2/images_thermal_{train,val}/coco.json`  
**Выход:** `datasets/person_flir/`

Кадры без людей по умолчанию пропускаются.

---

### `scripts/unpack_flir_thermal.sh`

**Зачем:** аккуратно распаковать из `archive.zip` только thermal (без огромного RGB).

**Выход:** `raw/FLIR_ADAS_v2/images_thermal_*` и `video_thermal_test`.

---

### `scripts/build_flir_video.py`

**Зачем:** FLIR хранит видео как кучу JPG. Скрипт склеивает один `video-id` в нормальный `.mp4`.

**Пример:**

```bash
python scripts/build_flir_video.py --video-id SCiKdG3MqZfiE292B --out demos/clip.mp4
```

---

## 6. Датасеты

| Датасет | Папка YOLO | Что внутри | Роль |
|---|---|---|---|
| LLVIP IR | `datasets/person_ir` | ~10582 train / 1443 val | первое обучение |
| FLIR thermal person | `datasets/person_flir` | ~8205 train / 819 val | второе обучение (дообучение) |
| FLIR `video_thermal_test` | только в `raw/` + `demos/` | отдельные ролики | **тест**, не обучение |

Формат метки YOLO (файл `.txt` рядом с картинкой):

```text
class_id x_center y_center width height
```

Все числа нормализованы от 0 до 1. У нас `class_id` всегда `0` (= person).

---

## 7. Как читать результаты обучения

Папка: `runs/detect/flir_thermal_yolov8n/` (или `llvip_ir_yolov8n`).

| Файл | Простыми словами |
|---|---|
| `results.png` | график «как училась»: ошибка падает, качество растёт |
| `results.csv` | те же цифры таблицей |
| `BoxPR_curve.png` | главный экзамен: нашла / не ошиблась (mAP50) |
| `BoxF1_curve.png` | лучший баланс порога confidence |
| `confusion_matrix*.png` | сколько людей нашла / пропустила |
| `val_batch*_pred.jpg` | примеры: что модель нарисовала |
| `val_batch*_labels.jpg` | примеры: как должно быть |
| `weights/best.pt` | лучшая версия модели |

Ориентиры по FLIR-модели (после дообучения):
- mAP50 ≈ **0.81**
- Precision ≈ **0.84**
- Recall ≈ **0.70**

Для просмотра видео удобный порог: **`CONF = 0.25` … `0.35`**.

---

## 8. Частые проблемы

| Проблема | Что делать |
|---|---|
| `CUDA out of memory` | `MODE = "low_vram"` |
| Окно `--show` не открывается | системный терминал + `opencv-python` (не headless) |
| На псевдоцвете почти нет детекций | `PREPROCESS = "whitehot"` |
| Камера не открывается | другой индекс `--source 1`, закрыть другие приложения |
| Хочу продолжить обучение | тот же `DATASET`, `MODE = "resume"` |
| Перепутал веса | для демо почти всегда `WEIGHTS = "flir"` |

---

## 9. Минимальный чеклист «я всё умею»

1. [ ] Активировал `.venv`
2. [ ] Знаю, где лежит актуальная модель: `weights/best_flir_thermal_yolov8n.pt`
3. [ ] Прогнал `eval_holdout_1.mp4` через `demo_track.py` и открыл `demo.mp4`
4. [ ] Понял, как в `train.py` переключать `DATASET` / `START` / `MODE`
5. [ ] Знаю, что для тепловизора с цветной палитрой нужен `whitehot`
6. [ ] Знаю, что `--source 0` — камера

---

## 10. Краткая шпаргалка команд

```bash
source .venv/bin/activate

# обучение кнопкой: открыть scripts/train.py → ▶

# видео → результат с рамками
python scripts/demo_track.py --source demos/eval_holdout_1.mp4 --conf 0.25 --name demo1

# живое окно (системный терминал)
python scripts/demo_track.py --source demos/eval_holdout_1.mp4 --show --no-save

# камера
python scripts/demo_track.py --source 0 --preprocess whitehot --show --no-save

# своё FLIR-видео из кадров
python scripts/build_flir_video.py --video-id ZAtDSNuZZjkZFvMAo --out demos/clip.mp4
```

---

*Документ соответствует упрощённой структуре проекта после очистки неиспользуемых папок и скриптов.*
