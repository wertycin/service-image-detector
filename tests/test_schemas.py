import pytest
from pydantic import ValidationError

from schemas import (
    BoundingBox,
    DetectionRequest,
    DetectionResponse,
    DetectionWithImageResponse,
    Point,
)


class TestPoint:

    def test_create_point(self):
        point = Point(x=100.5, y=200.3)
        assert point.x == 100.5
        assert point.y == 200.3

    def test_point_from_dict(self):
        point = Point.model_validate({"x": 50, "y": 75})
        assert point.x == 50
        assert point.y == 75


class TestBoundingBox:

    def test_create_bounding_box(self):
        bbox = BoundingBox(
            label="phone",
            confidence=0.95,
            top_left=Point(x=100, y=100),
            bottom_right=Point(x=200, y=200),
        )
        assert bbox.label == "phone"
        assert bbox.confidence == 0.95
        assert bbox.top_left.x == 100
        assert bbox.bottom_right.y == 200

    def test_confidence_validation(self):
        with pytest.raises(ValidationError):
            BoundingBox(
                label="phone",
                confidence=1.5,  # Некорректно
                top_left=Point(x=0, y=0),
                bottom_right=Point(x=100, y=100),
            )


class TestDetectionRequest:

    def test_request_with_url(self):
        request = DetectionRequest(url="https://example.com/image.jpg")
        assert request.url == "https://example.com/image.jpg"
        assert request.image_bytes is None

    def test_request_with_bytes(self):
        request = DetectionRequest(image_bytes="base64data")
        assert request.url is None
        assert request.image_bytes == "base64data"

    def test_request_with_bytes_alias(self):
        request = DetectionRequest.model_validate({"bytes": "base64data"})
        assert request.image_bytes == "base64data"

    def test_request_empty_fails(self):
        with pytest.raises(ValidationError) as exc_info:
            DetectionRequest()
        assert "must be provided" in str(exc_info.value)

    def test_request_both_fails(self):
        with pytest.raises(ValidationError) as exc_info:
            DetectionRequest(url="https://example.com/image.jpg", image_bytes="base64data")
        assert "not both" in str(exc_info.value)


class TestDetectionResponse:

    def test_empty_response(self):
        response = DetectionResponse()
        assert response.bboxes == []

    def test_response_with_bboxes(self):
        bbox = BoundingBox(
            label="phone",
            confidence=0.9,
            top_left=Point(x=0, y=0),
            bottom_right=Point(x=100, y=100),
        )
        response = DetectionResponse(bboxes=[bbox])
        assert len(response.bboxes) == 1
        assert response.bboxes[0].label == "phone"


class TestDetectionWithImageResponse:

    def test_response_with_image(self):
        bbox = BoundingBox(
            label="phone",
            confidence=0.9,
            top_left=Point(x=0, y=0),
            bottom_right=Point(x=100, y=100),
        )
        response = DetectionWithImageResponse(
            bboxes=[bbox],
            image="base64encodedimage",
        )
        assert len(response.bboxes) == 1
        assert response.image == "base64encodedimage"
