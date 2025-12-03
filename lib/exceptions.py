class DetectionBaseException(Exception):
    pass


class IncorrectInputFormat(DetectionBaseException):
    pass


class ImageDownloadError(DetectionBaseException):
    pass


class ModelInferenceError(DetectionBaseException):
    pass


class ImageProcessingError(DetectionBaseException):
    pass
