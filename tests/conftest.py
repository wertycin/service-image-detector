import base64
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

# Добавляем lib в путь для импортов
sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))


@pytest.fixture
def sample_image() -> np.ndarray:
    # Создаём RGB изображение 640x480 с некоторыми паттернами
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    # Добавляем цветные прямоугольники
    cv2.rectangle(image, (100, 100), (200, 200), (255, 0, 0), -1)
    cv2.rectangle(image, (300, 150), (400, 250), (0, 255, 0), -1)
    cv2.rectangle(image, (450, 200), (550, 350), (0, 0, 255), -1)
    return image


@pytest.fixture
def sample_image_bytes(sample_image: np.ndarray) -> bytes:
    _, encoded = cv2.imencode(".jpg", sample_image)
    return encoded.tobytes()


@pytest.fixture
def sample_image_base64(sample_image_bytes: bytes) -> str:
    return base64.b64encode(sample_image_bytes).decode("utf-8")


@pytest.fixture
def mock_detection_results():
    from clients.model import DetectionResult

    return [
        DetectionResult(
            class_id=0,
            label="phone",
            confidence=0.95,
            x1=100.0,
            y1=100.0,
            x2=200.0,
            y2=200.0,
        ),
        DetectionResult(
            class_id=1,
            label="link",
            confidence=0.87,
            x1=300.0,
            y1=150.0,
            x2=400.0,
            y2=250.0,
        ),
    ]


@pytest.fixture
def mock_model_client(mock_detection_results):
    from clients.model import ModelClient

    mock_client = MagicMock(spec=ModelClient)
    mock_client.is_loaded = True
    mock_client.predict.return_value = mock_detection_results
    mock_client.warmup.return_value = True
    return mock_client


@pytest.fixture
def mock_yolo_model():
    mock = MagicMock()

    # Мокаем результаты предсказания
    mock_boxes = MagicMock()
    mock_boxes.xyxy = [
        MagicMock(cpu=lambda: MagicMock(numpy=lambda: np.array([100, 100, 200, 200]))),
        MagicMock(cpu=lambda: MagicMock(numpy=lambda: np.array([300, 150, 400, 250]))),
    ]
    mock_boxes.conf = [
        MagicMock(cpu=lambda: MagicMock(numpy=lambda: 0.95)),
        MagicMock(cpu=lambda: MagicMock(numpy=lambda: 0.87)),
    ]
    mock_boxes.cls = [
        MagicMock(cpu=lambda: MagicMock(numpy=lambda: 0)),
        MagicMock(cpu=lambda: MagicMock(numpy=lambda: 1)),
    ]
    mock_boxes.__len__ = lambda self: 2

    mock_result = MagicMock()
    mock_result.boxes = mock_boxes

    mock.predict.return_value = [mock_result]

    return mock
