from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from clients.model import DetectionResult, ModelClient
from exceptions import ModelInferenceError


class TestDetectionResult:

    def test_create_detection_result(self):
        result = DetectionResult(
            class_id=0,
            label="phone",
            confidence=0.95,
            x1=100.0,
            y1=100.0,
            x2=200.0,
            y2=200.0,
        )
        assert result.class_id == 0
        assert result.label == "phone"
        assert result.confidence == 0.95
        assert result.x1 == 100.0


class TestModelClient:

    @pytest.fixture
    def client(self):
        with patch("clients.model.settings") as mock_settings:
            mock_settings.MODEL_PATH = "test_model.pt"
            mock_settings.DEVICE = "cpu"
            mock_settings.CONFIDENCE_THRESHOLD = 0.25
            mock_settings.IOU_THRESHOLD = 0.45
            return ModelClient()

    def test_init_default(self, client):
        assert client.model_path == "test_model.pt"
        assert client.device == "cpu"
        assert client.confidence_threshold == 0.25
        assert client.iou_threshold == 0.45
        assert client.model is None
        assert not client.is_loaded

    def test_resolve_device_cpu(self, client):
        result = client._resolve_device("cpu")
        assert result == "cpu"

    def test_resolve_device_cuda(self, client):
        result = client._resolve_device("cuda")
        assert result == "cuda"

    def test_resolve_device_auto_no_cuda(self, client):
        with patch.object(torch.cuda, "is_available", return_value=False):
            result = client._resolve_device("auto")
            assert result == "cpu"

    def test_resolve_device_auto_with_cuda(self, client):
        with patch.object(torch.cuda, "is_available", return_value=True):
            with patch.object(torch.cuda, "get_device_name", return_value="Test GPU"):
                result = client._resolve_device("auto")
                assert result == "cuda"

    def test_load_model_success(self, client):
        mock_model = MagicMock()

        with patch("clients.model.YOLO", return_value=mock_model):
            client.load_model()

            assert client.model is not None
            assert client.is_loaded
            mock_model.to.assert_called_once_with("cpu")

    def test_load_model_already_loaded(self, client):
        client._is_loaded = True
        client.model = MagicMock()

        with patch("clients.model.YOLO") as mock_yolo:
            client.load_model()
            mock_yolo.assert_not_called()

    def test_load_model_failure(self, client):
        with patch("clients.model.YOLO", side_effect=Exception("Load error")):
            with pytest.raises(ModelInferenceError) as exc_info:
                client.load_model()

            assert "Failed to load model" in str(exc_info.value)

    def test_warmup_success(self, client):
        mock_model = MagicMock()
        client.model = mock_model
        client._is_loaded = True

        result = client.warmup()

        assert result is True
        mock_model.predict.assert_called_once()

    def test_warmup_not_loaded(self, client):
        result = client.warmup()
        assert result is False

    def test_warmup_failure(self, client):
        mock_model = MagicMock()
        mock_model.predict.side_effect = Exception("Warmup error")
        client.model = mock_model
        client._is_loaded = True

        result = client.warmup()
        assert result is False

    def test_predict_not_loaded(self, client, sample_image):
        with pytest.raises(ModelInferenceError) as exc_info:
            client.predict(sample_image)

        assert "not loaded" in str(exc_info.value)

    def test_predict_success(self, client, sample_image):
        # Создаём правильную структуру мока
        mock_boxes = MagicMock()

        # Мокаем тензор-подобные объекты, которые возвращают numpy массивы
        xyxy_tensor_1 = MagicMock()
        xyxy_tensor_1.cpu.return_value.numpy.return_value = np.array([100.0, 100.0, 200.0, 200.0])

        xyxy_tensor_2 = MagicMock()
        xyxy_tensor_2.cpu.return_value.numpy.return_value = np.array([300.0, 150.0, 400.0, 250.0])

        conf_tensor_1 = MagicMock()
        conf_tensor_1.cpu.return_value.numpy.return_value = 0.95

        conf_tensor_2 = MagicMock()
        conf_tensor_2.cpu.return_value.numpy.return_value = 0.87

        cls_tensor_1 = MagicMock()
        cls_tensor_1.cpu.return_value.numpy.return_value = 0

        cls_tensor_2 = MagicMock()
        cls_tensor_2.cpu.return_value.numpy.return_value = 1

        mock_boxes.xyxy = [xyxy_tensor_1, xyxy_tensor_2]
        mock_boxes.conf = [conf_tensor_1, conf_tensor_2]
        mock_boxes.cls = [cls_tensor_1, cls_tensor_2]
        mock_boxes.__len__ = lambda self: 2

        mock_result = MagicMock()
        mock_result.boxes = mock_boxes

        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_result]

        client.model = mock_model
        client._is_loaded = True

        results = client.predict(sample_image)

        assert len(results) == 2
        assert results[0].class_id == 0
        assert results[0].label == "phone"
        assert results[0].confidence == 0.95
        assert results[1].class_id == 1
        assert results[1].label == "link"

    def test_predict_no_detections(self, client, sample_image):
        mock_result = MagicMock()
        mock_result.boxes = None

        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_result]

        client.model = mock_model
        client._is_loaded = True

        results = client.predict(sample_image)

        assert len(results) == 0
