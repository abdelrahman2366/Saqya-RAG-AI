# Handoff: plant-disease detection from photos

**Goal:** a farmer sends a photo of a sick plant. SAQYA says what it most likely is, with honest uncertainty,
and gives treatment advice from the bulletins, with citations.

Set up the project first: [`ONBOARDING.md`](../ONBOARDING.md).

## What is already done

The whole path around the model is built and tested (`backend/tests/test_vision.py`). You add the model, then the
chat UI.

```
photo ─► POST /vision/diagnose ─► check_image (type, size, readable)
       ─► DiseaseDetector.predict  ◄── YOUR MODEL GOES HERE
       ─► parse_label ("Tomato___Late_blight" → crop tomato, disease late blight)
       ─► confidence ≥ VISION_MIN_CONFIDENCE and not healthy?
            yes → question "How do I control late blight on tomato?" → normal RAG answer (cited advice)
            no  → "not sure, send a clearer photo" / "looks healthy" (no treatment advice)
       ─► response: predictions + message + advice
```

| file | what it does |
|---|---|
| `backend/app/services/vision.py` | `DiseaseDetector` interface, `HFImageClassifier` (loads any Hugging Face image-classification model from `models/`), `parse_label`, `check_image`, `VisionService` (confidence rule, advice question in Arabic/English), farmer-facing messages |
| `backend/app/api/routes/vision.py` | `GET /vision/status`, `POST /vision/diagnose` (multipart: `file`, optional `crop`, `language` ar/en, `advice`) |
| `backend/app/schemas/vision.py` | response models (`DiagnosisResponse`, `PredictionOut`, `VisionStatus`) |
| `backend/app/core/config.py` | `VISION_ENABLED`, `VISION_MODEL`, `VISION_DEVICE`, `VISION_ALLOW_DOWNLOAD`, `VISION_MIN_CONFIDENCE`, `VISION_TOP_K`, `VISION_MAX_MB` |
| `frontend/api_client.py` | `ApiClient.diagnose_image(filename, data, mime, crop, language, advice)`, ready for the UI |
| `backend/tests/test_vision.py` | tests with a fake detector: label parsing, upload checks, confident / uncertain / healthy, disabled mode |

Off by default. With `VISION_ENABLED=false`, `/vision/diagnose` answers 503 and nothing else in the app changes.

## Your tasks

### 1. Choose the data and the classes
- Start from SAQYA's crops: tomato, potato, pepper, cucumber, onion, wheat, maize, rice, citrus, mango, grape,
  date palm… (`GET /library` lists all). A disease the bulletins cover can be explained with cited advice; one
  they don't cover still goes through the web / general-knowledge fallback.
- Public datasets to consider:
  - **PlantVillage**: lab photos of single leaves on plain backgrounds, many classes, high accuracy, but weak on
    real field photos.
  - **PlantDoc**: real-world field photos, harder and closer to what farmers send.

  Check each dataset's licence before using it, and record it in your model folder.
- Always keep a **healthy** class per crop, and consider an "unknown / not a plant" class.
- Expect a domain gap: farmers' phone photos (Egyptian fields, sun, dust, several leaves) differ from lab
  images. Collect even 100–300 real photos for testing if you can.

### 2. Train or pick a model
- Easiest path: fine-tune an image-classification backbone (ViT, ConvNeXt, EfficientNet, MobileNetV3 for speed)
  with Hugging Face `transformers`/`timm`. Save it with `save_pretrained` into `models/plant-disease/`, including
  `config.json` with `id2label` and `preprocessor_config.json`.
- Name labels `Crop___Disease` (or `Crop___healthy`) so `parse_label` reads them, or adjust `parse_label`.
- Keep a proper split. Split **by source photo / plant**, not by augmented copy, so near-identical images aren't
  in both train and test. Choose the epoch and hyper-parameters on validation, then report test once.
- Your own architecture (e.g. a detector that finds lesions first) works too. Write a class with `name`,
  `available()` and `predict(image_bytes, top_k) -> list[Prediction]`, and return it from `load_detector()` in
  `vision.py`.

### 3. Switch it on and check it
Add these lines to `backend\.env`, then restart the backend:
```
VISION_ENABLED=true
VISION_MODEL=models/plant-disease
```
- `GET http://localhost:8000/vision/status` should show `enabled: true`. The model loads on the first photo.
- Try it at `http://localhost:8000/docs` → `POST /vision/diagnose` → "Try it out" → upload a leaf photo.

### 4. Set `VISION_MIN_CONFIDENCE` from data (don't guess)
The default 0.7 is a placeholder. On your **validation** photos, find the confidence at which predictions are
right often enough, e.g. ≥ 90% precision. Report accuracy, macro-F1, per-class recall, and expected calibration
error on the **test** photos. `backend/app/evaluation/metrics.py` already has `expected_calibration_error`,
`brier_score`, `auroc` and `risk_coverage`. Put the numbers in `docs/ai-ml-engineering.md` (a new section),
marking anything not measured as NOT EVALUATED.

### 5. Add it to the chat UI (`frontend/app.py`)
- Add a photo button next to the document attach control. Send the photo with
  `api.diagnose_image(name, bytes, mime, crop=..., language=...)`.
- Show `message` first (it always says it is a suggestion, not a confirmed diagnosis), then the top predictions
  with their percentages. If `advice` is present, render it like a normal answer (`advice.answer`,
  `advice.sources`).
- Arabic and English: the UI already switches direction. Use `ui_text.py` for the new labels in both languages.
- Don't change unrelated parts of the UI.

### 6. Tests and pull request
- Keep `backend/tests/test_vision.py` passing. Add tests for anything you change (e.g. a new label format).
- `python -m pytest -q` must pass. Open a pull request with the evaluation numbers.

## Safety rules (from the project's engineering rules)
- A photo prediction is never presented as a confirmed diagnosis. Low confidence means no treatment advice, and
  the farmer is asked for a clearer photo.
- Treatment advice comes from the bulletins through the normal RAG pipeline, with its grounding and pesticide
  safety checks. The vision code never writes doses itself.
- Models load lazily and never download at startup. Model files stay out of git (`models/` is ignored), so share
  them like the vector store (a release asset or a shared drive).
- Keep the feature off by default until its test numbers justify switching it on.

## Definition of done
- [ ] Model in `models/plant-disease/` (or a custom detector), with its licence and training notes
- [ ] Test-split metrics (accuracy, macro-F1, per-class recall, ECE) in `docs/ai-ml-engineering.md`
- [ ] `VISION_MIN_CONFIDENCE` chosen on validation data
- [ ] Photo button in the chat app, Arabic and English
- [ ] Tests pass, PR opened
