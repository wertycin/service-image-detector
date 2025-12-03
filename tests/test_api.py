from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from clients.model import DetectionResult
from exceptions import ImageDownloadError, IncorrectInputFormat, ModelInferenceError
from flow.flow import DetectionFlow, DetectionFlowResult
from schemas import BoundingBox, DetectionResponse, DetectionWithImageResponse, Point


@pytest.fixture
def mock_detection_results():
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
    mock_client.load_model.return_value = None
    return mock_client


@pytest.fixture
def mock_flow_result(mock_detection_results):
    return DetectionFlowResult(
        detections=mock_detection_results,
        annotated_image_base64="base64annotatedimage",
    )


@pytest.fixture
def mock_detection_flow(mock_flow_result, mock_detection_results):
    mock = MagicMock(spec=DetectionFlow)

    mock.process = AsyncMock(return_value=mock_flow_result)
    mock.to_detection_response.return_value = DetectionResponse(
        bboxes=[
            BoundingBox(
                label="phone",
                confidence=0.95,
                top_left=Point(x=100.0, y=100.0),
                bottom_right=Point(x=200.0, y=200.0),
            )
        ]
    )
    mock.to_detection_with_image_response.return_value = DetectionWithImageResponse(
        bboxes=[
            BoundingBox(
                label="phone",
                confidence=0.95,
                top_left=Point(x=100.0, y=100.0),
                bottom_right=Point(x=200.0, y=200.0),
            )
        ],
        image="base64annotatedimage",
    )
    return mock


@pytest.fixture
def test_client(mock_model_client, mock_detection_flow):
    import main

    # Переопределяем зависимости
    main.app.dependency_overrides[main.get_model_client] = lambda: mock_model_client
    main.app.dependency_overrides[main.get_detection_flow] = lambda: mock_detection_flow

    with TestClient(main.app, raise_server_exceptions=False) as client:
        yield client, mock_model_client, mock_detection_flow

    # Очищаем переопределения
    main.app.dependency_overrides.clear()


class TestHealthEndpoint:

    def test_health_check(self, test_client):
        client, mock_model_client, _ = test_client

        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True


class TestForwardEndpoint:

    def test_forward_with_url(self, test_client):
        client, _, mock_detection_flow = test_client

        response = client.post(
            "/forward",
            json={"url": "https://example.com/image.jpg"},
        )

        assert response.status_code == 200
        mock_detection_flow.process.assert_called_once()

    def test_forward_with_bytes(self, test_client):
        client, _, mock_detection_flow = test_client

        response = client.post(
            "/forward",
            json={"bytes": "base64imagedata"},
        )

        assert response.status_code == 200
        mock_detection_flow.process.assert_called_once()

    def test_forward_empty_request(self, test_client):
        client, _, _ = test_client

        response = client.post("/forward", json={})

        assert response.status_code == 400
        assert response.json()["detail"] == "bad request"

    def test_forward_both_inputs(self, test_client):
        client, _, _ = test_client

        response = client.post(
            "/forward",
            json={
                "url": "https://example.com/image.jpg",
                "bytes": "base64data",
            },
        )

        assert response.status_code == 400
        assert response.json()["detail"] == "bad request"

    def test_forward_model_error(self, mock_model_client):
        import main

        mock_flow = MagicMock(spec=DetectionFlow)
        mock_flow.process = AsyncMock(side_effect=ModelInferenceError("Model failed"))

        main.app.dependency_overrides[main.get_model_client] = lambda: mock_model_client
        main.app.dependency_overrides[main.get_detection_flow] = lambda: mock_flow

        with TestClient(main.app, raise_server_exceptions=False) as client:
            response = client.post(
                "/forward",
                json={"url": "https://example.com/image.jpg"},
            )

            assert response.status_code == 403
            assert "модель не смогла обработать данные" in response.json()["detail"]

        main.app.dependency_overrides.clear()

    def test_forward_download_error(self, mock_model_client):
        import main

        mock_flow = MagicMock(spec=DetectionFlow)
        mock_flow.process = AsyncMock(side_effect=ImageDownloadError("Download failed"))

        main.app.dependency_overrides[main.get_model_client] = lambda: mock_model_client
        main.app.dependency_overrides[main.get_detection_flow] = lambda: mock_flow

        with TestClient(main.app, raise_server_exceptions=False) as client:
            response = client.post(
                "/forward",
                json={"url": "https://example.com/image.jpg"},
            )

            assert response.status_code == 400
            assert response.json()["detail"] == "bad request"

        main.app.dependency_overrides.clear()

    def test_forward_incorrect_input_error(self, mock_model_client):
        import main

        mock_flow = MagicMock(spec=DetectionFlow)
        mock_flow.process = AsyncMock(side_effect=IncorrectInputFormat("Bad input"))

        main.app.dependency_overrides[main.get_model_client] = lambda: mock_model_client
        main.app.dependency_overrides[main.get_detection_flow] = lambda: mock_flow

        with TestClient(main.app, raise_server_exceptions=False) as client:
            response = client.post(
                "/forward",
                json={"url": "https://example.com/image.jpg"},
            )

            assert response.status_code == 400
            assert response.json()["detail"] == "bad request"

        main.app.dependency_overrides.clear()


class TestForwardWithShowEndpoint:

    def test_forward_with_show_url(self, test_client):
        client, _, mock_detection_flow = test_client

        response = client.post(
            "/forward_with_show",
            json={"url": "https://example.com/image.jpg"},
        )

        assert response.status_code == 200
        mock_detection_flow.process.assert_called_once()
        # Проверяем, что визуализация была запрошена
        call_kwargs = mock_detection_flow.process.call_args[1]
        assert call_kwargs.get("with_visualization") is True

    def test_forward_with_show_bytes(self, test_client):
        client, _, _ = test_client

        response = client.post(
            "/forward_with_show",
            json={"bytes": "base64imagedata"},
        )

        assert response.status_code == 200

    def test_forward_with_show_empty_request(self, test_client):
        client, _, _ = test_client

        response = client.post("/forward_with_show", json={})

        assert response.status_code == 400
        assert response.json()["detail"] == "bad request"

    def test_forward_with_show_model_error(self, mock_model_client):
        import main

        mock_flow = MagicMock(spec=DetectionFlow)
        mock_flow.process = AsyncMock(side_effect=ModelInferenceError("Model failed"))

        main.app.dependency_overrides[main.get_model_client] = lambda: mock_model_client
        main.app.dependency_overrides[main.get_detection_flow] = lambda: mock_flow

        with TestClient(main.app, raise_server_exceptions=False) as client:
            response = client.post(
                "/forward_with_show",
                json={"url": "https://example.com/image.jpg"},
            )

            assert response.status_code == 403

        main.app.dependency_overrides.clear()
