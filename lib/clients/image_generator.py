import base64
import logging

import cv2
import numpy as np

from clients.model import DetectionResult
from consts import VisualizationConfig, get_class_color
from exceptions import ImageProcessingError

logger = logging.getLogger(__name__)


class ImageGeneratorClient:

    def __init__(self, config: VisualizationConfig | None = None):
        self.config = config or VisualizationConfig()

    def draw_detections(
        self,
        image: np.ndarray,
        detections: list[DetectionResult],
    ) -> np.ndarray:
        logger.debug(f"Drawing {len(detections)} detections on image")

        # Создаём копию, чтобы не модифицировать оригинальное изображение
        annotated = image.copy()

        for detection in detections:
            self._draw_single_detection(annotated, detection)

        return annotated

    def _draw_single_detection(
        self,
        image: np.ndarray,
        detection: DetectionResult,
    ) -> None:
        # Получаем координаты как целые числа
        x1, y1 = int(detection.x1), int(detection.y1)
        x2, y2 = int(detection.x2), int(detection.y2)

        # Получаем цвет для этого класса
        color = get_class_color(detection.class_id)

        # Рисуем ограничивающую рамку
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            color,
            self.config.BBOX_THICKNESS,
        )

        # Подготавливаем текст метки
        label_text = f"{detection.label}: {detection.confidence:.2f}"

        # Вычисляем размер текста
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            self.config.FONT_SCALE,
            self.config.FONT_THICKNESS,
        )

        # Вычисляем позицию фона метки
        label_y1 = max(0, y1 - text_height - 2 * self.config.LABEL_PADDING)
        label_y2 = y1
        label_x1 = x1
        label_x2 = x1 + text_width + 2 * self.config.LABEL_PADDING

        # Рисуем фон метки
        cv2.rectangle(
            image,
            (label_x1, label_y1),
            (label_x2, label_y2),
            color,
            -1,  # Залитый прямоугольник
        )

        # Рисуем текст метки
        text_x = label_x1 + self.config.LABEL_PADDING
        text_y = label_y2 - self.config.LABEL_PADDING

        cv2.putText(
            image,
            label_text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.config.FONT_SCALE,
            self.config.LABEL_TEXT_COLOR,
            self.config.FONT_THICKNESS,
        )

    def encode_to_base64(
        self,
        image: np.ndarray,
        format: str = ".jpg",
        quality: int = 95,
    ) -> str:
        logger.debug(f"Encoding image to base64 with format {format}")

        try:
            encode_params = []
            if format.lower() in [".jpg", ".jpeg"]:
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
            elif format.lower() == ".png":
                encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 6]

            success, encoded = cv2.imencode(format, image, encode_params)

            if not success:
                raise ImageProcessingError("Failed to encode image")

            base64_string = base64.b64encode(encoded.tobytes()).decode("utf-8")
            logger.debug(f"Encoded image to base64 ({len(base64_string)} chars)")

            return base64_string

        except Exception as e:
            if isinstance(e, ImageProcessingError):
                raise
            logger.error(f"Failed to encode image: {e}")
            raise ImageProcessingError(f"Failed to encode image: {str(e)}")

    def generate_annotated_base64(
        self,
        image: np.ndarray,
        detections: list[DetectionResult],
        format: str = ".jpg",
        quality: int = 95,
    ) -> str:
        annotated = self.draw_detections(image, detections)
        return self.encode_to_base64(annotated, format, quality)
