"""Plant-disease detection extension point: label parsing, upload checks, confidence rule, hand-off to RAG.
No model is needed: a fake detector stands in for the one a team member will add."""

import io

import pytest
from PIL import Image

from app.main import app
from app.services.rag_service import RAGService
from app.services.retrieval import RetrievedChunk
from app.services.vision import (NotConfigured, Prediction, VisionService, VisionUnavailable, check_image,
                                 BadImage, parse_label)
from tests.conftest import FakeGenerator, FakeRetriever, make_client

BLIGHT = RetrievedChunk("tomato_0007", "Late blight on tomato is controlled by removing infected leaves and spraying "
                        "a recommended fungicide.", "tomato.pdf", 7, 7, 0.82, "vegetables / tomato", "Diseases")


def png(size=(32, 32)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (40, 160, 60)).save(buf, format="PNG")
    return buf.getvalue()


class FakeDetector:
    name = "fake-plant-model"

    def __init__(self, predictions):
        self.predictions = predictions

    def available(self):
        return True

    def predict(self, image, top_k=3):
        return self.predictions[:top_k]


@pytest.mark.parametrize("label, expected", [
    ("Tomato___Late_blight", ("tomato", "late blight", False)),
    ("Corn_(maize)___Common_rust_", ("corn", "common rust", False)),
    ("Apple___healthy", ("apple", "", True)),
    ("Tomato___Tomato_Yellow_Leaf_Curl_Virus", ("tomato", "yellow leaf curl virus", False)),
    ("tomato late blight", ("tomato", "late blight", False)),
])
def test_parse_label(label, expected):
    assert parse_label(label) == expected


def test_image_checks():
    check_image(png(), "image/png", 10)
    with pytest.raises(BadImage):
        check_image(b"", "image/png", 10)
    with pytest.raises(BadImage):
        check_image(b"not an image at all", "image/jpeg", 10)
    with pytest.raises(BadImage):
        check_image(png(), "application/pdf", 10)
    with pytest.raises(BadImage):
        check_image(png((600, 600)), "image/png", 0.0001)


def test_not_configured_says_how_to_enable():
    with pytest.raises(VisionUnavailable, match="computer-vision-handoff"):
        NotConfigured().predict(png())


def _client(predictions, min_confidence=0.7):
    rag = RAGService(FakeRetriever([BLIGHT]), FakeGenerator(), top_k=4, min_similarity=0.3, translate_queries=False)
    client = make_client(rag)
    app.state.vision = VisionService(FakeDetector(predictions), min_confidence=min_confidence)
    return client


def _post(client, data=None, mime="image/png", **form):
    return client.post("/vision/diagnose", files={"file": ("leaf.png", data if data is not None else png(), mime)},
                       data={"language": "en", **form})


def test_confident_disease_gets_advice_from_the_bulletins():
    client = _client([Prediction("Tomato___Late_blight", 0.93, "tomato", "late blight"),
                      Prediction("Tomato___Early_blight", 0.05, "tomato", "early blight")])
    body = _post(client).json()
    assert body["confident"] and body["top"]["disease"] == "late blight" and len(body["predictions"]) == 2
    assert "not a confirmed diagnosis" in body["message"]
    assert body["advice"]["sources"] == ["[1] tomato.pdf, p. 7"]


def test_uncertain_or_healthy_photo_gets_no_treatment_advice():
    unsure = _post(_client([Prediction("Tomato___Late_blight", 0.41, "tomato", "late blight")])).json()
    assert not unsure["confident"] and unsure["advice"] is None and "not sure" in unsure["message"]
    healthy = _post(_client([Prediction("Tomato___healthy", 0.97, "tomato", "", True)])).json()
    assert healthy["confident"] and healthy["advice"] is None and "healthy" in healthy["message"]


def test_bad_uploads_and_disabled_vision():
    client = _client([Prediction("Tomato___Late_blight", 0.9, "tomato", "late blight")])
    assert _post(client, data=b"").status_code == 400
    assert _post(client, mime="application/pdf").status_code == 415
    app.state.vision = VisionService(NotConfigured())
    response = _post(client)
    assert response.status_code == 503 and "computer-vision-handoff" in response.json()["detail"]
    assert client.get("/vision/status").json()["enabled"] is False


def test_advice_question_uses_glossary_names_in_arabic():
    q = VisionService.advice_question(Prediction("Tomato___Late_blight", 0.9, "tomato", "late blight"), "ar")
    assert "الطماطم" in q and q.endswith("؟")
    assert VisionService.advice_question(Prediction("x", 0.9, "", "rust"), "en") == "How do I control rust?"
