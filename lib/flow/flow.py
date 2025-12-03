import logging
from dataclasses import dataclass, field

import numpy as np

from clients.image_downloader import ImageDownloaderClient
from clients.image_generator import ImageGeneratorClient
from clients.model import DetectionResult, ModelClient
from exceptions import IncorrectInputFormat, ModelInferenceError
from schemas import BoundingBox, DetectionResponse, DetectionWithImageResponse, Point

logger = logging.getLogger(__name__)


@dataclass
class DetectionFlowResult:
    detections: list[DetectionResult] = field(default_factory=list)
    image: np.ndarray | None = None
    annotated_image_base64: str | None = None


class DetectionFlow:

    def __init__(
        self,
        model_client: ModelClient,
        image_downloader: ImageDownloaderClient | None = None,
        image_generator: ImageGeneratorClient | None = None,
    ):
        self.model_client = model_client
        self.image_downloader = image_downloader or ImageDownloaderClient()
        self.image_generator = image_generator or ImageGeneratorClient()

    async def process(
        self,
        url: str | None = None,
        image_bytes: str | None = None,
        with_visualization: bool = False,
    ) -> DetectionFlowResult:
        logger.info(
            f"Processing detection request: url={url is not None}, "
            f"bytes={image_bytes is not None}, visualization={with_visualization}"
        )

        # Валидация входных данных
        if url is None and image_bytes is None:
            raise IncorrectInputFormat("Either 'url' or 'bytes' must be provided")
        if url is not None and image_bytes is not None:
            raise IncorrectInputFormat("Only one of 'url' or 'bytes' should be provided")

        # Шаг 1: Получение изображения
        image = await self._acquire_image(url, image_bytes)

        # Шаг 2: Запуск инференса
        try:
            detections = self.model_client.predict(image)
        except ModelInferenceError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error during inference: {e}")
            raise ModelInferenceError("модель не смогла обработать данные")

        # Шаг 3: Генерация картинки с bboex, если было запрошено
        result = DetectionFlowResult(detections=detections, image=image)

        if with_visualization:
            result.annotated_image_base64 = self.image_generator.generate_annotated_base64(
                image, detections
            )

        logger.info(f"Detection completed: {len(detections)} objects found")
        return result

    async def _acquire_image(
        self,
        url: str | None,
        image_bytes: str | None,
    ) -> np.ndarray:
        if url is not None:
            return await self.image_downloader.download_from_url(url)
        else:
            return self.image_downloader.decode_from_base64(image_bytes)

    def to_detection_response(
        self,
        result: DetectionFlowResult,
    ) -> DetectionResponse:
        bboxes = self._convert_detections_to_bboxes(result.detections)
        return DetectionResponse(bboxes=bboxes)

    def to_detection_with_image_response(
        self,
        result: DetectionFlowResult,
    ) -> DetectionWithImageResponse:
        bboxes = self._convert_detections_to_bboxes(result.detections)
        return DetectionWithImageResponse(
            bboxes=bboxes,
            image=result.annotated_image_base64 or "",
        )

    def _convert_detections_to_bboxes(
        self,
        detections: list[DetectionResult],
    ) -> list[BoundingBox]:
        return [
            BoundingBox(
                label=det.label,
                confidence=det.confidence,
                top_left=Point(x=det.x1, y=det.y1),
                bottom_right=Point(x=det.x2, y=det.y2),
            )
            for det in detections
        ]
