from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from clients.model import DetectionResult
from exceptions import IncorrectInputFormat, ModelInferenceError
from flow.flow import DetectionFlow, DetectionFlowResult


class TestDetectionFlowResult:

    def test_default_values(self):
        result = DetectionFlowResult()
        assert result.detections == []
        assert result.image is None
        assert result.annotated_image_base64 is None

    def test_with_values(self, mock_detection_results, sample_image):
        result = DetectionFlowResult(
            detections=mock_detection_results,
            image=sample_image,
            annotated_image_base64="base64data",
        )
        assert len(result.detections) == 2
        assert result.image is not None
        assert result.annotated_image_base64 == "base64data"


class TestDetectionFlow:

    @pytest.fixture
    def mock_image_downloader(self, sample_image):
        from clients.image_downloader import ImageDownloaderClient

        mock = MagicMock(spec=ImageDownloaderClient)
        mock.download_from_url = AsyncMock(return_value=sample_image)
        mock.decode_from_base64 = MagicMock(return_value=sample_image)
        return mock

    @pytest.fixture
    def mock_image_generator(self):
        from clients.image_generator import ImageGeneratorClient

        mock = MagicMock(spec=ImageGeneratorClient)
        mock.generate_annotated_base64.return_value = "annotated_base64"
        return mock

    @pytest.fixture
    def flow(self, mock_model_client, mock_image_downloader, mock_image_generator):
        return DetectionFlow(
            model_client=mock_model_client,
            image_downloader=mock_image_downloader,
            image_generator=mock_image_generator,
        )

    def test_init(self, mock_model_client):
        flow = DetectionFlow(model_client=mock_model_client)
        assert flow.model_client is not None
        assert flow.image_downloader is not None
        assert flow.image_generator is not None

    @pytest.mark.asyncio
    async def test_process_with_url(self, flow, sample_image):
        result = await flow.process(url="https://example.com/image.jpg")

        assert len(result.detections) == 2
        assert result.annotated_image_base64 is None
        flow.image_downloader.download_from_url.assert_called_once_with(
            "https://example.com/image.jpg"
        )

    @pytest.mark.asyncio
    async def test_process_with_bytes(self, flow):
        result = await flow.process(image_bytes="base64data")

        assert len(result.detections) == 2
        flow.image_downloader.decode_from_base64.assert_called_once_with("base64data")

    @pytest.mark.asyncio
    async def test_process_with_visualization(self, flow):
        result = await flow.process(
            url="https://example.com/image.jpg",
            with_visualization=True,
        )

        assert result.annotated_image_base64 == "annotated_base64"
        flow.image_generator.generate_annotated_base64.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_no_input_fails(self, flow):
        with pytest.raises(IncorrectInputFormat) as exc_info:
            await flow.process()

        assert "must be provided" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_process_both_inputs_fails(self, flow):
        with pytest.raises(IncorrectInputFormat) as exc_info:
            await flow.process(
                url="https://example.com/image.jpg",
                image_bytes="base64data",
            )

        assert "should be provided" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_process_model_error(
        self, mock_model_client, mock_image_downloader, mock_image_generator
    ):
        mock_model_client.predict.side_effect = ModelInferenceError("Model failed")

        flow = DetectionFlow(
            model_client=mock_model_client,
            image_downloader=mock_image_downloader,
            image_generator=mock_image_generator,
        )

        with pytest.raises(ModelInferenceError):
            await flow.process(url="https://example.com/image.jpg")

    def test_to_detection_response(self, flow, mock_detection_results):
        flow_result = DetectionFlowResult(detections=mock_detection_results)
        response = flow.to_detection_response(flow_result)

        assert len(response.bboxes) == 2
        assert response.bboxes[0].label == "phone"
        assert response.bboxes[0].confidence == 0.95
        assert response.bboxes[0].top_left.x == 100.0

    def test_to_detection_with_image_response(self, flow, mock_detection_results):
        flow_result = DetectionFlowResult(
            detections=mock_detection_results,
            annotated_image_base64="base64image",
        )
        response = flow.to_detection_with_image_response(flow_result)

        assert len(response.bboxes) == 2
        assert response.image == "base64image"

    def test_convert_detections_to_bboxes(self, flow, mock_detection_results):
        bboxes = flow._convert_detections_to_bboxes(mock_detection_results)

        assert len(bboxes) == 2
        assert bboxes[0].label == "phone"
        assert bboxes[0].confidence == 0.95
        assert bboxes[0].top_left.x == 100.0
        assert bboxes[0].top_left.y == 100.0
        assert bboxes[0].bottom_right.x == 200.0
        assert bboxes[0].bottom_right.y == 200.0
