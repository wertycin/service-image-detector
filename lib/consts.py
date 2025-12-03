import os
from enum import Enum


class DetectorClasses(Enum):
    PHONE = 0
    LINK = 1
    EMAIL = 2
    LOGIN = 3
    QR = 4

    @classmethod
    def get_label(cls, class_id: int) -> str:
        for member in cls:
            if member.value == class_id:
                return member.name.lower()
        return f"class_{class_id}"

    @classmethod
    def get_all_labels(cls) -> list[str]:
        return [member.name.lower() for member in cls]


class Settings:
    MODEL_PATH: str = os.getenv("MODEL_PATH", "data/models/yolo_contacts.pt")
    DEVICE: str = os.getenv("DEVICE", "auto")  # 'auto', 'cpu', 'cuda', 'cuda:0' и т.д.
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.25"))
    IOU_THRESHOLD: float = float(os.getenv("IOU_THRESHOLD", "0.45"))
    IMAGE_DOWNLOAD_TIMEOUT: int = int(os.getenv("IMAGE_DOWNLOAD_TIMEOUT", "30"))
    MAX_IMAGE_SIZE: int = int(os.getenv("MAX_IMAGE_SIZE", "10485760"))  # 10 МБ


settings = Settings()


# Настройки визуализации для рисования bboxes
class VisualizationConfig:
    BBOX_COLOR = (0, 255, 0)  # Зелёный в BGR
    BBOX_THICKNESS = 2
    FONT_SCALE = 0.6
    FONT_THICKNESS = 2
    LABEL_BACKGROUND_COLOR = (0, 255, 0)  # Зелёный фон для метки
    LABEL_TEXT_COLOR = (0, 0, 0)  # Чёрный текст
    LABEL_PADDING = 5


# Цветовая палитра для разных классов (формат BGR)
CLASS_COLORS = {
    DetectorClasses.PHONE: (255, 0, 0),    # Синий
    DetectorClasses.LINK: (0, 255, 0),     # Зелёный
    DetectorClasses.EMAIL: (0, 0, 255),    # Красный
    DetectorClasses.LOGIN: (255, 255, 0),  # Голубой
    DetectorClasses.QR: (255, 0, 255),     # Пурпурный
}


def get_class_color(class_id: int) -> tuple[int, int, int]:
    for cls, color in CLASS_COLORS.items():
        if cls.value == class_id:
            return color
    return (128, 128, 128)  # Серый для неизвестных классов (такого быть не должно)
