"""Common model interface and input / output contract for every model the platform serves.

Every model implements:
    load_model()            load and integrity-check the artifact (raises on any mismatch; never substitutes)
    preprocess(inp)         decode + validate + transform the input exactly as in training
    predict(inp)            PredictionOutput (label, probability, threshold, details, timings)
    predict_proba(X)        probabilities for already-preprocessed model inputs (patch / feature level)
    get_metadata()          architecture, version, training data, execution mode, input contract
    get_metrics()           recorded evaluation results with their split and evaluation unit

Input contract (ModelInput): the bytes of one image (PNG, JPEG, uncompressed DICOM, NIfTI) and optional adjacent
slices (prev / next, prev2 / next2) for 2.5-D and 5-slice models. Output contract (PredictionOutput): one slice-level
call per request in {"Tumour detected", "No tumour detected", "Uncertain / unable to classify"} or a withheld /
needs-context status with the reason, plus the model's probability (calibrated only where the card says so).
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

LABELS = ("Tumour detected", "No tumour detected", "Uncertain / unable to classify")
BAND = 0.10


class ModelUnavailable(RuntimeError):
    """The model cannot run (missing artifact, failed integrity check, wrong environment). Never faked."""


class InputRejected(ValueError):
    def __init__(self, status: int, message: str, detail: dict | None = None):
        super().__init__(message)
        self.status, self.message, self.detail = status, message, detail or {}


@dataclass
class ModelInput:
    data: bytes
    filename: str | None = None
    prev: bytes | None = None
    next: bytes | None = None
    prev2: bytes | None = None
    next2: bytes | None = None


@dataclass
class PredictionOutput:
    model_id: str
    model_version: str
    status: str                       # ok | withheld | needs_context | rejected
    label: str | None
    probability: float | None
    threshold: float | None
    execution_mode: str
    dataset_version: str | None
    probability_note: str = ""
    reason: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    timings_ms: dict[str, float] = field(default_factory=dict)
    images: dict[str, bytes] = field(default_factory=dict)     # name -> PNG bytes (preview, patch map, Grad-CAM)
    explanation: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def slice_call(pmax: float, threshold: float, band: float = BAND) -> str:
    if pmax >= threshold + band:
        return LABELS[0]
    if pmax < threshold - band:
        return LABELS[1]
    return LABELS[2]


class BaseModel(abc.ABC):
    model_id: str = ""
    version: str = ""

    @abc.abstractmethod
    def load_model(self) -> None: ...

    @abc.abstractmethod
    def preprocess(self, inp: ModelInput) -> Any: ...

    @abc.abstractmethod
    def predict(self, inp: ModelInput) -> PredictionOutput: ...

    def predict_proba(self, X):
        raise NotImplementedError(f"{self.model_id} does not expose feature-level probabilities")

    @abc.abstractmethod
    def get_metadata(self) -> dict: ...

    @abc.abstractmethod
    def get_metrics(self) -> dict: ...
