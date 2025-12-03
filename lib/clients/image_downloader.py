import base64
import logging

import cv2
import httpx
import numpy as np

from consts import settings
from exceptions import ImageDownloadError, ImageProcessingError

logger = logging.getLogger(__name__)


class ImageDownloaderClient:

    def __init__(self, timeout: int | None = None):
        self.timeout = timeout or settings.IMAGE_DOWNLOAD_TIMEOUT
        self.max_size = settings.MAX_IMAGE_SIZE

    async def download_from_url(self, url: str) -> np.ndarray:
        logger.info(f"Downloading image from URL: {url}")

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url)
                response.raise_for_status()

                content_length = len(response.content)
                if content_length > self.max_size:
                    raise ImageDownloadError(
                        f"Image size ({content_length} bytes) exceeds maximum allowed ({self.max_size} bytes)"
                    )

                return self._decode_image_bytes(response.content)

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error while downloading image: {e}")
            raise ImageDownloadError(f"Failed to download image: HTTP {e.response.status_code}")
        except httpx.RequestError as e:
            logger.error(f"Request error while downloading image: {e}")
            raise ImageDownloadError(f"Failed to download image: {str(e)}")

    def decode_from_base64(self, base64_string: str) -> np.ndarray:
        logger.debug("Decoding image from base64")

        try:
            # Обработка схемы data URI, если присутствует
            if "," in base64_string:
                base64_string = base64_string.split(",", 1)[1]

            image_bytes = base64.b64decode(base64_string)

            if len(image_bytes) > self.max_size:
                raise ImageProcessingError(
                    f"Image size ({len(image_bytes)} bytes) exceeds maximum allowed ({self.max_size} bytes)"
                )

            return self._decode_image_bytes(image_bytes)

        except Exception as e:
            if isinstance(e, ImageProcessingError):
                raise
            logger.error(f"Failed to decode base64 image: {e}")
            raise ImageProcessingError(f"Failed to decode base64 image: {str(e)}")

    def _decode_image_bytes(self, image_bytes: bytes) -> np.ndarray:
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                raise ImageProcessingError("Failed to decode image: invalid image format")

            logger.debug(f"Successfully decoded image with shape: {image.shape}")
            return image

        except Exception as e:
            if isinstance(e, ImageProcessingError):
                raise
            logger.error(f"Failed to decode image bytes: {e}")
            raise ImageProcessingError(f"Failed to decode image: {str(e)}")
