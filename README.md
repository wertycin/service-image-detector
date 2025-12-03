# service-image-detection

ML микросервис для детекции объектов на изображении с помощью моделей от Ultralitics YOLO.

## Ручки

- **POST /forward** - обнаружить объекты на фото, возвращает список bboxes
- **POST /forward_with_show** - обнаружить объекты и вернуть изображение с отрисованными bboxes
- **GET /health** - health check

## Установка окружения

### С помощью uv (recommended)

```bash
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### С помощью pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Запуск сервиса

### Локально

```bash
PYTHONPATH=lib uvicorn main:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
docker-compose up --build
```

## API

### Детекция (JSON response)

```bash
# With URL
curl -X POST http://localhost:8000/forward \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/image.jpg"}'

# With base64 bytes
curl -X POST http://localhost:8000/forward \
  -H "Content-Type: application/json" \
  -d '{"bytes": "<base64_encoded_image>"}'
```

### Детекция с отрисовкой bboxes

```bash
curl -X POST http://localhost:8000/forward_with_show \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/image.jpg"}'
```

## Формат ответа

### /forward response

```json
{
  "bboxes": [
    {
      "label": "phone",
      "confidence": 0.95,
      "top_left": {"x": 100.0, "y": 100.0},
      "bottom_right": {"x": 200.0, "y": 200.0}
    }
  ]
}
```

### /forward_with_show response

```json
{
  "bboxes": [...],
  "image": "<base64_encoded_annotated_image>"
}
```

## UI
После запуска сервиса (локально или в Docker) открыть ссылку
```txt
http://localhost:8000/
```

## Коды возвращаемых ошибок

- **400** - Bad request (invalid input format)
- **403** - Model processing error

## Поддерживаемые классы для детекции

- PHONE (0)
- LINK (1)
- EMAIL (2)
- LOGIN (3)
- QR (4)

## Конфигурация

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| MODEL_PATH | data/models/yolo_contacts.pt | Path to YOLO model |
| DEVICE | auto | Device (auto/cpu/cuda) |
| LOG_LEVEL | INFO | Logging level |
| CONFIDENCE_THRESHOLD | 0.25 | Detection confidence threshold |
| IOU_THRESHOLD | 0.45 | NMS IoU threshold |
| IMAGE_DOWNLOAD_TIMEOUT | 30 | Image download timeout (seconds) |
| MAX_IMAGE_SIZE | 10485760 | Max image size (bytes) |

## Запуск тестов

```bash
# Локально
pytest tests/ -v
```

```bash
# Docker
uv pip install --system pytest pytest-asyncio
pytest tests/ -v
```
