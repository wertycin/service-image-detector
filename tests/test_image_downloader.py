import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import httpx
import numpy as np

from clients.image_downloader import ImageDownloaderClient
from exceptions import ImageDownloadError, ImageProcessingError


class TestImageDownloaderClient:

    @pytest.fixture
    def client(self):
        return ImageDownloaderClient(timeout=10)

    def test_init_default(self):
        client = ImageDownloaderClient()
        assert client.timeout > 0
        assert client.max_size > 0

    def test_init_custom_timeout(self):
        client = ImageDownloaderClient(timeout=60)
        assert client.timeout == 60

    @pytest.mark.asyncio
    async def test_download_from_url_success(
        self, client, sample_image_bytes
    ):
        mock_response = MagicMock()
        mock_response.content = sample_image_bytes
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await client.download_from_url("https://example.com/image.jpg")

            assert isinstance(result, np.ndarray)
            assert len(result.shape) == 3  # Height, width, channels

    @pytest.mark.asyncio
    async def test_download_from_url_http_error(self, client):
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Not Found",
            request=MagicMock(),
            response=mock_response,
        )

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

            with pytest.raises(ImageDownloadError) as exc_info:
                await client.download_from_url("https://example.com/notfound.jpg")

            assert "404" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_download_from_url_size_exceeded(self, client, sample_image_bytes):
        client.max_size = 100  # Set very small limit

        mock_response = MagicMock()
        mock_response.content = sample_image_bytes
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

            with pytest.raises(ImageDownloadError) as exc_info:
                await client.download_from_url("https://example.com/large.jpg")

            assert "exceeds maximum" in str(exc_info.value)

    def test_decode_from_base64_success(self, client, sample_image_base64):
        result = client.decode_from_base64(sample_image_base64)

        assert isinstance(result, np.ndarray)
        assert len(result.shape) == 3

    def test_decode_from_base64_with_data_uri(self, client, sample_image_base64):
        data_uri = f"data:image/jpeg;base64,{sample_image_base64}"
        result = client.decode_from_base64(data_uri)

        assert isinstance(result, np.ndarray)

    def test_decode_from_base64_invalid(self, client):
        with pytest.raises(ImageProcessingError):
            client.decode_from_base64("not-valid-base64!!!")

    def test_decode_image_bytes_invalid(self, client):
        with pytest.raises(ImageProcessingError) as exc_info:
            client._decode_image_bytes(b"not an image")

        assert "invalid image format" in str(exc_info.value)
