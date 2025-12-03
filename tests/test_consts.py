import pytest

from consts import DetectorClasses, Settings, VisualizationConfig, get_class_color


class TestDetectorClasses:

    def test_class_values(self):
        assert DetectorClasses.PHONE.value == 0
        assert DetectorClasses.LINK.value == 1
        assert DetectorClasses.EMAIL.value == 2
        assert DetectorClasses.LOGIN.value == 3
        assert DetectorClasses.QR.value == 4

    def test_get_label_known_class(self):
        assert DetectorClasses.get_label(0) == "phone"
        assert DetectorClasses.get_label(1) == "link"
        assert DetectorClasses.get_label(2) == "email"
        assert DetectorClasses.get_label(3) == "login"
        assert DetectorClasses.get_label(4) == "qr"

    def test_get_label_unknown_class(self):
        result = DetectorClasses.get_label(99)
        assert result == "class_99"

    def test_get_all_labels(self):
        labels = DetectorClasses.get_all_labels()
        assert len(labels) == 5
        assert "phone" in labels
        assert "link" in labels
        assert "email" in labels
        assert "login" in labels
        assert "qr" in labels


class TestSettings:

    def test_default_values(self):
        settings = Settings()
        assert settings.MODEL_PATH.endswith("data/models/yolo_contacts.pt")
        assert settings.DEVICE in ("auto", "cpu", "cuda")
        assert 0 < settings.CONFIDENCE_THRESHOLD <= 1.0
        assert 0 < settings.IOU_THRESHOLD <= 1.0
        assert settings.IMAGE_DOWNLOAD_TIMEOUT > 0
        assert settings.MAX_IMAGE_SIZE > 0


class TestVisualizationConfig:

    def test_default_values(self):
        config = VisualizationConfig()
        assert config.BBOX_COLOR == (0, 255, 0)
        assert config.BBOX_THICKNESS == 2
        assert config.FONT_SCALE == 0.6


class TestGetClassColor:

    def test_known_class_colors(self):
        assert get_class_color(0) == (255, 0, 0)  # PHONE - Синий
        assert get_class_color(1) == (0, 255, 0)  # LINK - Зелёный
        assert get_class_color(2) == (0, 0, 255)  # EMAIL - Красный
        assert get_class_color(3) == (255, 255, 0)  # LOGIN - Голубой
        assert get_class_color(4) == (255, 0, 255)  # QR - Пурпурный

    def test_unknown_class_color(self):
        result = get_class_color(99)
        assert result == (128, 128, 128)  # Серый
