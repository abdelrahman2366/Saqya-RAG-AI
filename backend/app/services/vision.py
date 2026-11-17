"""Plant-disease detection from a photo — the extension point for SAQYA's computer-vision feature.

Status: the plumbing is complete and tested (API route, upload checks, label parsing, confidence rule, the hand-off
to the RAG pipeline for treatment advice); **no disease model ships with the project yet**. Off by default
(VISION_ENABLED=false). docs/computer-vision-handoff.md explains how to add a model.

How a photo is handled (POST /vision/diagnose):
1. the image is checked (type, size, readable) and passed to a `DiseaseDetector`;
2. the detector returns its top-k labels with probabilities;
3. each label is parsed into crop / disease / healthy (PlantVillage-style "Tomato___Late_blight" and
   "tomato late blight" both work);
4. if the best label is a disease with confidence >= VISION_MIN_CONFIDENCE, SAQYA builds a question such as
   "How do I control late blight on tomato?" and answers it with the normal RAG pipeline, so the advice comes
   from the bulletins with citations. Below the threshold, or for a healthy leaf, no advice is fetched and the
   farmer is asked for a clearer photo: a photo model's guess is never presented as a confirmed diagnosis.

Two ways to plug a model in:
- **No code:** save a Hugging Face image-classification model (fine-tuned ViT, ConvNeXt, EfficientNet...) to
  models/plant-disease/ (config.json with id2label, preprocessor_config.json, weights) and set VISION_ENABLED=true.
  `HFImageClassifier` below loads it lazily, once, from disk only (VISION_ALLOW_DOWNLOAD=true to fetch by id).
- **Own code:** write a class with `name`, `available()` and `predict(image_bytes, top_k)` (see `DiseaseDetector`)
  and return it from `load_detector()`. Everything else (route, checks, advice, tests) stays the same.
"""

from __future__ import annotations

import io
import logging
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Protocol

logger = logging.getLogger(__name__)

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
_HEALTHY = re.compile(r"(?i)\bhealthy\b|سليم")


class VisionUnavailable(RuntimeError):
    """No disease model is configured or it could not be loaded."""


class BadImage(ValueError):
    """The upload is not a readable image."""


@dataclass
class Prediction:
    label: str                  # the model's own label
    confidence: float           # probability, 0..1
    crop: str = ""              # parsed from the label ("tomato")
    disease: str = ""           # parsed from the label ("late blight"); "" when healthy
    healthy: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Diagnosis:
    predictions: list[Prediction] = field(default_factory=list)
    model: str = ""
    latency_ms: float = 0.0

    @property
    def top(self) -> Prediction | None:
        return self.predictions[0] if self.predictions else None


class DiseaseDetector(Protocol):
    """What the rest of SAQYA needs from a disease model."""

    name: str

    def available(self) -> bool: ...

    def predict(self, image: bytes, top_k: int = 3) -> list[Prediction]: ...


def parse_label(label: str) -> tuple[str, str, bool]:
    """(crop, disease, healthy) from a class label. Handles PlantVillage "Tomato___Late_blight",
    "Corn_(maize)___Common_rust_", "Apple___healthy" and plain "tomato late blight"."""
    text = label.strip()
    if "___" in text:
        crop, _, rest = text.partition("___")
    elif "__" in text:
        crop, _, rest = text.partition("__")
    else:
        parts = text.replace("_", " ").split()
        crop, rest = (parts[0], " ".join(parts[1:])) if parts else ("", "")
    crop = re.sub(r"\(.*?\)", "", crop.replace("_", " ")).strip(" ,").lower()
    rest = re.sub(r"\s+", " ", rest.replace("_", " ")).strip(" ,").lower()
    healthy = bool(_HEALTHY.search(rest) or _HEALTHY.search(text))
    disease = "" if healthy else rest
    # "Tomato___Tomato_Yellow_Leaf_Curl_Virus" -> "yellow leaf curl virus"
    if disease.startswith(crop + " "):
        disease = disease[len(crop) + 1:]
    return crop, disease, healthy


def check_image(data: bytes, content_type: str | None, max_mb: float) -> None:
    """Raise BadImage for an empty, too large, wrongly typed or unreadable upload."""
    if not data:
        raise BadImage("The photo is empty.")
    if len(data) > max_mb * 1024 * 1024:
        raise BadImage(f"The photo is larger than {max_mb:g} MB.")
    if content_type and content_type.split(";")[0].strip().lower() not in ALLOWED_TYPES:
        raise BadImage("Send a JPEG, PNG or WebP photo.")
    try:
        from PIL import Image

        with Image.open(io.BytesIO(data)) as img:
            img.verify()
    except Exception as exc:  # noqa: BLE001
        raise BadImage("The photo could not be read.") from exc


class NotConfigured:
    name = "none"

    def __init__(self, reason: str = "VISION_ENABLED is false") -> None:
        self.reason = reason

    def available(self) -> bool:
        return False

    def predict(self, image: bytes, top_k: int = 3) -> list[Prediction]:
        raise VisionUnavailable(f"No plant-disease model: {self.reason}. See docs/computer-vision-handoff.md.")


class HFImageClassifier:
    """A Hugging Face image-classification model from a local folder, loaded on first use (once, thread-safe)."""

    def __init__(self, model: str, device: str = "auto", allow_download: bool = False) -> None:
        self.name = model
        self.device = device
        self.allow_download = allow_download
        self._pipe = None
        self._failed: str | None = None
        self._lock = threading.Lock()

    @property
    def failed(self) -> str | None:
        return self._failed

    def _load(self):
        if self._pipe is not None or self._failed:
            return self._pipe
        with self._lock:
            if self._pipe is not None or self._failed:
                return self._pipe
            try:
                from transformers import pipeline

                from app.services.reranker import local_model_path, pick_device

                location = local_model_path(self.name)
                if location is None and not self.allow_download:
                    raise FileNotFoundError(f"model folder '{self.name}' not found (put it in models/ or set "
                                            "VISION_ALLOW_DOWNLOAD=true for a Hugging Face id)")
                device = pick_device(self.device)
                self._pipe = pipeline("image-classification", model=location or self.name,
                                      device=0 if device == "cuda" else -1)
                logger.info("Plant-disease model %s loaded on %s", self.name, device)
            except Exception as exc:  # noqa: BLE001
                self._failed = f"{type(exc).__name__}: {exc}"[:300]
                logger.warning("Plant-disease model could not be loaded: %s", self._failed)
        return self._pipe

    def available(self) -> bool:
        return self._load() is not None

    def predict(self, image: bytes, top_k: int = 3) -> list[Prediction]:
        pipe = self._load()
        if pipe is None:
            raise VisionUnavailable(self._failed or "model not loaded")
        from PIL import Image

        with Image.open(io.BytesIO(image)) as img:
            rgb = img.convert("RGB")
        out = []
        for item in pipe(rgb, top_k=top_k):
            crop, disease, healthy = parse_label(str(item["label"]))
            out.append(Prediction(str(item["label"]), round(float(item["score"]), 4), crop, disease, healthy))
        return out


def load_detector(settings) -> DiseaseDetector:
    """The detector the settings ask for. Nothing is loaded here: a model loads on the first photo."""
    if not settings.vision_enabled:
        return NotConfigured()
    return HFImageClassifier(settings.vision_model, settings.vision_device, settings.vision_allow_download)


class VisionService:
    def __init__(self, detector: DiseaseDetector, min_confidence: float = 0.7, top_k: int = 3) -> None:
        self.detector = detector
        self.min_confidence = min_confidence
        self.top_k = top_k

    def diagnose(self, image: bytes) -> Diagnosis:
        started = time.perf_counter()
        predictions = self.detector.predict(image, self.top_k)
        predictions = sorted(predictions, key=lambda p: p.confidence, reverse=True)[: self.top_k]
        return Diagnosis(predictions, getattr(self.detector, "name", ""), round((time.perf_counter() - started) * 1000, 1))

    def confident(self, diagnosis: Diagnosis) -> bool:
        top = diagnosis.top
        return top is not None and top.confidence >= self.min_confidence

    @staticmethod
    def advice_question(prediction: Prediction, language: str, crop_hint: str | None = None) -> str:
        """The question handed to the RAG pipeline for a detected disease (crop and disease names translated with
        the glossary when it knows them)."""
        from app.services.glossary import GLOSSARY

        crop = (crop_hint or prediction.crop or "").strip()
        disease = prediction.disease
        if language == "ar":
            names = {t.en.lower(): t.ar for t in GLOSSARY}
            crop_ar = names.get(crop.lower(), crop)
            disease_ar = names.get(disease.lower(), disease)
            return f"كيف أكافح مرض {disease_ar} في {crop_ar}؟" if crop_ar else f"كيف أكافح مرض {disease_ar}؟"
        return f"How do I control {disease} on {crop}?" if crop else f"How do I control {disease}?"


NOTE = {
    "en": "This is a suggestion from one photo, not a confirmed diagnosis. Check the plant closely or ask your "
          "local extension officer before treating.",
    "ar": "هذا اقتراح من صورة واحدة وليس تشخيصًا مؤكدًا. افحص النبات جيدًا أو استشر المرشد الزراعي قبل المعالجة.",
}
UNCERTAIN = {
    "en": "I'm not sure what this is from the photo. Try a clear, close photo of one affected leaf in daylight, "
          "and tell me the crop.",
    "ar": "لست متأكدة مما يظهر في الصورة. جرّب صورة واضحة وقريبة لورقة مصابة واحدة في ضوء النهار، واذكر لي المحصول.",
}
HEALTHY = {
    "en": "The leaf looks healthy in this photo. If you still see a problem, send a photo of the affected part.",
    "ar": "تبدو الورقة سليمة في هذه الصورة. إذا كنت ما زلت ترى مشكلة، أرسل صورة للجزء المصاب.",
}
