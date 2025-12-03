import base64

import cv2
import numpy as np
import pytest

from clients.image_generator import ImageGeneratorClient
from clients.model import DetectionResult
from consts import VisualizationConfig


class TestImageGeneratorClient:

    @pytest.fixture
    def client(self):
        return ImageGeneratorClient()

    @pytest.fixture
    def detections(self):
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

    def test_init_default(self):
        client = ImageGeneratorClient()
        assert client.config is not None

    def test_init_custom_config(self):
        config = VisualizationConfig()
        config.BBOX_THICKNESS = 5
        client = ImageGeneratorClient(config=config)
        assert client.config.BBOX_THICKNESS == 5

    def test_draw_detections(self, client, sample_image, detections):
        result = client.draw_detections(sample_image, detections)

        # Результат должен иметь ту же форму, что и вход
        assert result.shape == sample_image.shape

        # Оригинал не должен быть изменён
        assert not np.array_equal(result, sample_image)

    def test_draw_detections_empty(self, client, sample_image):
        result = client.draw_detections(sample_image, [])

        # Должен вернуть копию оригинала
        assert np.array_equal(result, sample_image)

    def test_draw_single_detection(self, client, sample_image, detections):
        original = sample_image.copy()
        client._draw_single_detection(sample_image, detections[0])

        # Изображение должно быть модифицировано на месте
        assert not np.array_equal(sample_image, original)

    def test_encode_to_base64_jpg(self, client, sample_image):
        result = client.encode_to_base64(sample_image, format=".jpg")

        # Should be valid base64
        decoded = base64.b64decode(result)
        assert len(decoded) > 0

        # Должен декодироваться обратно в изображение
        nparr = np.frombuffer(decoded, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        assert img is not None
        assert img.shape[0] > 0

    def test_encode_to_base64_png(self, client, sample_image):
        result = client.encode_to_base64(sample_image, format=".png")

        decoded = base64.b64decode(result)
        assert len(decoded) > 0

    def test_encode_to_base64_quality(self, client, sample_image):
        high_quality = client.encode_to_base64(sample_image, format=".jpg", quality=95)
        low_quality = client.encode_to_base64(sample_image, format=".jpg", quality=10)

        # Высокое качество должно давать больший размер вывода
        assert len(high_quality) > len(low_quality)

    def test_generate_annotated_base64(self, client, sample_image, detections):
        result = client.generate_annotated_base64(sample_image, detections)

        # Должен быть валидным base64
        decoded = base64.b64decode(result)
        assert len(decoded) > 0

        # Должен декодироваться
        nparr = np.frombuffer(decoded, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        assert img is not None
