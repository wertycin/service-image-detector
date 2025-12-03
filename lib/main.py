import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from clients.model import ModelClient
from consts import settings
from exceptions import (
    ImageDownloadError,
    ImageProcessingError,
    IncorrectInputFormat,
    ModelInferenceError,
)
from flow.flow import DetectionFlow
from schemas import (
    DetectionRequest,
    DetectionResponse,
    DetectionWithImageResponse,
    ErrorResponse,
    HealthResponse,
)

# Настройка логирования
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class AppState:
    model_client: ModelClient | None = None
    detection_flow: DetectionFlow | None = None


app_state = AppState()


def get_model_client() -> ModelClient:
    if app_state.model_client is None:
        raise RuntimeError("Model client not initialized")
    return app_state.model_client


def get_detection_flow() -> DetectionFlow:
    if app_state.detection_flow is None:
        raise RuntimeError("Detection flow not initialized")
    return app_state.detection_flow


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting image detection service...")

    # Инициализация клиента модели
    app_state.model_client = ModelClient()

    try:
        app_state.model_client.load_model()
        app_state.model_client.warmup()
    except ModelInferenceError as e:
        logger.error(f"Failed to initialize model: {e}")
        raise

    # Инициализация флоу детекции
    app_state.detection_flow = DetectionFlow(model_client=app_state.model_client)

    logger.info("Service started successfully")

    yield

    # Очистка
    logger.info("Shutting down service...")


app = FastAPI(
    title="Image Detection Service",
    description="ML microservice for object detection using YOLO",
    version="0.1.0",
    lifespan=lifespan,
)

# Путь к статическим файлам
STATIC_DIR = Path(__file__).parent.parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# Ручка для UI
@app.get("/", include_in_schema=False)
async def root():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "Image Detection Service"}


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(f"Validation error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": "bad request"},
    )


@app.exception_handler(ValidationError)
async def pydantic_validation_exception_handler(request: Request, exc: ValidationError):
    logger.warning(f"Validation error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": "bad request"},
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check endpoint",
)
async def health_check(
    model_client: Annotated[ModelClient, Depends(get_model_client)]
) -> HealthResponse:
    return HealthResponse(
        status="healthy",
        model_loaded=model_client.is_loaded,
    )


@app.post(
    "/forward",
    response_model=DetectionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad request"},
        403: {"model": ErrorResponse, "description": "Model processing error"},
    },
    summary="Detect objects in image",
)
async def detect(
    request: DetectionRequest,
    detection_flow: Annotated[DetectionFlow, Depends(get_detection_flow)],
) -> DetectionResponse:
    try:
        result = await detection_flow.process(
            url=request.url,
            image_bytes=request.image_bytes,
            with_visualization=False,
        )
        return detection_flow.to_detection_response(result)

    except IncorrectInputFormat:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="bad request",
        )
    except (ImageDownloadError, ImageProcessingError) as e:
        logger.warning(f"Image processing error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="bad request",
        )
    except ModelInferenceError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="модель не смогла обработать данные",
        )


@app.post(
    "/forward_with_show",
    response_model=DetectionWithImageResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad request"},
        403: {"model": ErrorResponse, "description": "Model processing error"},
    },
    summary="Detect objects and return annotated image",
)
async def show_detection(
    request: DetectionRequest,
    detection_flow: Annotated[DetectionFlow, Depends(get_detection_flow)],
) -> DetectionWithImageResponse:
    try:
        result = await detection_flow.process(
            url=request.url,
            image_bytes=request.image_bytes,
            with_visualization=True,
        )
        return detection_flow.to_detection_with_image_response(result)

    except IncorrectInputFormat:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="bad request",
        )
    except (ImageDownloadError, ImageProcessingError) as e:
        logger.warning(f"Image processing error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="bad request",
        )
    except ModelInferenceError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="модель не смогла обработать данные",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
