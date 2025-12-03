import logging
from dataclasses import dataclass

import numpy as np
import torch
from ultralytics import YOLO

from consts import DetectorClasses, settings
from exceptions import ModelInferenceError

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    class_id: int
    label: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


class ModelClient:

    def __init__(
        self,
        model_path: str | None = None,
        device: str | None = None,
        confidence_threshold: float | None = None,
        iou_threshold: float | None = None,
    ):
        self.model_path = model_path or settings.MODEL_PATH
        self.device = self._resolve_device(device or settings.DEVICE)
        self.confidence_threshold = confidence_threshold or settings.CONFIDENCE_THRESHOLD
        self.iou_threshold = iou_threshold or settings.IOU_THRESHOLD
        self.model: YOLO | None = None
        self._is_loaded = False

    def _resolve_device(self, device: str) -> str:
        if device == "auto":
            if torch.cuda.is_available():
                resolved = "cuda"
                logger.info(f"CUDA available, using GPU: {torch.cuda.get_device_name(0)}")
            else:
                resolved = "cpu"
                logger.info("CUDA not available, using CPU")
            return resolved
        return device

    def load_model(self) -> None:
        if self._is_loaded:
            logger.debug("Model already loaded")
            return

        logger.info(f"Loading model from {self.model_path} on device {self.device}")

        try:
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            self._is_loaded = True
            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise ModelInferenceError(f"Failed to load model: {str(e)}")

    def warmup(self, image_size: tuple[int, int] = (640, 640)) -> bool:
        if not self._is_loaded:
            logger.warning("Model not loaded, cannot perform warmup")
            return False

        logger.info("Performing model warmup...")

        try:
            dummy_image = np.zeros((*image_size, 3), dtype=np.uint8)
            self.model.predict(
                dummy_image,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False,
            )
            logger.info("Model warmup completed successfully")
            return True

        except Exception as e:
            logger.error(f"Model warmup failed: {e}")
            return False

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def predict(self, image: np.ndarray) -> list[DetectionResult]:
        if not self._is_loaded:
            raise ModelInferenceError("Model not loaded. Call load_model() first.")

        logger.debug(f"Running inference on image with shape: {image.shape}")

        try:
            results = self.model.predict(
                image,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False,
            )

            detections = []

            for result in results:
                if result.boxes is None:
                    continue

                boxes = result.boxes
                for i in range(len(boxes)):
                    # Получаем координаты рамки (формат xyxy)
                    box = boxes.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = box

                    # Получаем уверенность и класс
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls_id = int(boxes.cls[i].cpu().numpy())

                    # Получаем метку из нашего enum или имён модели
                    label = DetectorClasses.get_label(cls_id)

                    detection = DetectionResult(
                        class_id=cls_id,
                        label=label,
                        confidence=conf,
                        x1=float(x1),
                        y1=float(y1),
                        x2=float(x2),
                        y2=float(y2),
                    )
                    detections.append(detection)

            logger.debug(f"Found {len(detections)} detections")
            return detections

        except Exception as e:
            if isinstance(e, ModelInferenceError):
                raise
            logger.error(f"Inference failed: {e}")
            raise ModelInferenceError(f"Model inference failed: {str(e)}")
