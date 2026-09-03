"""Production-oriented FastAPI inference service for EEG ensembles."""

from __future__ import annotations

import hmac
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field, model_validator

from eeg_project.logging_config import log_event, setup_logging
from inference import load_deployment_predictor

logger = setup_logging(json_logs=os.getenv("EEG_JSON_LOGS", "true").lower() == "true")
predictor = None
startup_error: str | None = None
DEFAULT_MANIFEST = "artifacts/production/manifest.json"
MAX_BATCH_SIZE = int(os.getenv("EEG_MAX_BATCH_SIZE", "64"))
MAX_SAMPLES = int(os.getenv("EEG_MAX_SAMPLES", "5000"))
MAX_CONCURRENT_INFERENCES = int(os.getenv("EEG_MAX_CONCURRENT_INFERENCES", "1"))
INFERENCE_SEMAPHORE = threading.BoundedSemaphore(max(1, MAX_CONCURRENT_INFERENCES))


@dataclass
class ServiceMetrics:
    requests_total: int = 0
    predictions_total: int = 0
    prediction_errors_total: int = 0
    overload_total: int = 0
    prediction_latency_seconds_total: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def record_request(self) -> None:
        with self.lock:
            self.requests_total += 1

    def record_prediction(self, elapsed_seconds: float, failed: bool = False) -> None:
        with self.lock:
            self.predictions_total += 1
            self.prediction_latency_seconds_total += elapsed_seconds
            self.prediction_errors_total += int(failed)

    def record_overload(self) -> None:
        with self.lock:
            self.overload_total += 1

    def render_prometheus(self) -> str:
        with self.lock:
            average = self.prediction_latency_seconds_total / self.predictions_total if self.predictions_total else 0.0
            lines = {
                "eeg_http_requests_total": self.requests_total,
                "eeg_prediction_requests_total": self.predictions_total,
                "eeg_prediction_errors_total": self.prediction_errors_total,
                "eeg_inference_overload_total": self.overload_total,
                "eeg_prediction_latency_seconds_total": self.prediction_latency_seconds_total,
                "eeg_prediction_latency_seconds_average": average,
            }
        return "\n".join(f"{name} {value}" for name, value in lines.items()) + "\n"


metrics = ServiceMetrics()


@asynccontextmanager
async def lifespan(_: FastAPI):
    global predictor, startup_error
    manifest = os.getenv("EEG_MIXED_ENSEMBLE_MANIFEST", DEFAULT_MANIFEST)
    require_model = os.getenv("EEG_REQUIRE_MODEL", "false").lower() == "true"
    try:
        if not Path(manifest).is_file():
            raise FileNotFoundError(f"Model manifest does not exist: {manifest}")
        predictor = load_deployment_predictor(
            manifest,
            verify_integrity=os.getenv("EEG_VERIFY_ARTIFACTS", "true").lower() == "true",
        )
        startup_error = None
        log_event(logger, "model_loaded", manifest=manifest, members=predictor.ensemble_size)
    except Exception as error:
        predictor = None
        startup_error = str(error)
        log_event(logger, "model_load_failed", error_type=type(error).__name__)
        if require_model:
            raise
    yield


app = FastAPI(
    title="EEG Learning State API",
    version="2.0.0",
    lifespan=lifespan,
)

origins = [origin.strip() for origin in os.getenv("EEG_CORS_ORIGINS", "").split(",") if origin.strip()]
if origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-API-Key", "X-Request-ID"],
    )


class PredictionRequest(BaseModel):
    samples: list[list[list[float]]] = Field(description="EEG batch shaped [N, channels, time]")
    tta: bool = Field(default=False, description="Enable only after validating TTA for the deployed model")

    @model_validator(mode="after")
    def validate_batch(self) -> "PredictionRequest":
        if not self.samples:
            raise ValueError("samples must contain at least one record")
        if len(self.samples) > MAX_BATCH_SIZE:
            raise ValueError(f"batch size exceeds limit of {MAX_BATCH_SIZE}")
        channels = len(self.samples[0])
        points = len(self.samples[0][0]) if channels else 0
        if channels == 0 or points == 0:
            raise ValueError("each EEG record must contain non-empty channels and samples")
        if channels * points > MAX_SAMPLES:
            raise ValueError(f"sample size exceeds limit of {MAX_SAMPLES} points per record")
        for record in self.samples:
            if len(record) != channels or any(len(channel) != points for channel in record):
                raise ValueError("all records must have the same [channels, time] shape")
        return self


def require_api_key(request: Request) -> None:
    configured_key = os.getenv("EEG_API_KEY")
    if not configured_key:
        return
    supplied_key = request.headers.get("X-API-Key", "")
    if not hmac.compare_digest(supplied_key, configured_key):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid API key")


@app.middleware("http")
async def request_logging(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    started = time.perf_counter()
    metrics.record_request()
    try:
        response = await call_next(request)
    except Exception:
        duration_ms = (time.perf_counter() - started) * 1000
        log_event(
            logger,
            "http_request_failed",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            duration_ms=round(duration_ms, 1),
        )
        raise
    duration_ms = (time.perf_counter() - started) * 1000
    log_event(
        logger,
        "http_request",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        duration_ms=round(duration_ms, 1),
    )
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = f"{duration_ms:.1f}"
    return response


@app.get("/live")
def live():
    return {"status": "alive"}


@app.get("/health")
@app.get("/ready")
def health():
    if predictor is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Model is not ready")
    return {"status": "ready", "ensemble_size": predictor.ensemble_size}


@app.get("/metadata", dependencies=[Depends(require_api_key)])
def metadata():
    if predictor is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Model is not configured")
    return {
        "api_version": app.version,
        "ensemble_size": predictor.ensemble_size,
        "classes": [str(label) for label in predictor.classes],
        "input_shape": list(predictor.input_shape),
        "method": predictor.method,
        "temperature": predictor.temperature,
    }


@app.get("/metrics", dependencies=[Depends(require_api_key)], response_class=PlainTextResponse)
def prometheus_metrics():
    return metrics.render_prometheus()


@app.post("/predict", dependencies=[Depends(require_api_key)])
def predict(request: PredictionRequest):
    if predictor is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Model is not configured")
    if request.tta and os.getenv("EEG_ALLOW_TTA", "false").lower() != "true":
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "TTA is disabled for this deployment")
    if not INFERENCE_SEMAPHORE.acquire(blocking=False):
        metrics.record_overload()
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Inference capacity is temporarily exhausted")
    started = time.perf_counter()
    try:
        samples = np.asarray(request.samples, dtype=np.float32)
        predictions = predictor.predict(samples, tta=request.tta)
        latency_seconds = time.perf_counter() - started
        metrics.record_prediction(latency_seconds)
        return {
            "predictions": predictions,
            "latency_ms": round(latency_seconds * 1000, 2),
        }
    except ValueError as error:
        metrics.record_prediction(time.perf_counter() - started, failed=True)
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(error)) from error
    except Exception as error:
        metrics.record_prediction(time.perf_counter() - started, failed=True)
        log_event(logger, "inference_failed", error_type=type(error).__name__)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Inference failed") from error
    finally:
        INFERENCE_SEMAPHORE.release()
