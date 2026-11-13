"""Egyptian Arabic: the hand-written Egyptian test questions, the extra dialect words, and loading the
Egyptian fine-tune of bge-m3 (models/bge-m3-egyptian, made by notebooks/egyptian_finetune.ipynb)."""

import json
from pathlib import Path

import pytest

from app.services.dialect import is_egyptian, to_msa
from app.services.retrieval import ModelNotFoundError, resolve_model

DATA = Path(__file__).resolve().parents[2] / "data"


def test_egyptian_test_questions_mirror_the_standard_ones():
    egyptian = json.loads((DATA / "eval_questions_egyptian.json").read_text(encoding="utf-8"))
    standard = {q["id"]: q for q in json.loads((DATA / "eval_questions.json").read_text(encoding="utf-8"))}
    assert [q["id"] for q in egyptian] == [f"e{n:02d}" for n in range(1, 56)]
    for q in egyptian:
        original = standard[q["msa_id"]]
        assert q["question_msa"] == original["question"]
        assert q["expected_keywords"] == original["expected_keywords"]
        assert q.get("expected_source") == original.get("expected_source")
        assert is_egyptian(q["question"]), q["question"]
        assert q["question"] != original["question"]


@pytest.mark.parametrize("spoken, standard", [
    ("الفدان ياخد كام كيلو بذرة عباد الشمس؟", "الفدان يحتاج كم كيلو بذرة دوار الشمس؟"),
    ("ليه لازم أحط خلايا نحل جنب غيط عباد الشمس؟", "لماذا لازم أحط خلايا نحل بجوار حقل دوار الشمس؟"),
    ("أعرف إزاي إن البطيخة استوت؟", "أعرف كيف إن البطيخ ناضج؟"),
    ("ليه البرتقال بيفرقع على الشجر؟", "لماذا الموالح يتشقق على الأشجار؟"),
    ("الشتلة الكويسة بتاعة الخضار شكلها إيه؟", "الشتلة الجيدة الخضر شكلها ما؟"),
])
def test_more_egyptian_words(spoken, standard):
    assert to_msa(spoken) == standard


def test_standard_test_questions_are_left_alone():
    for q in json.loads((DATA / "eval_questions.json").read_text(encoding="utf-8"))[:55]:
        assert to_msa(q["question"]) == q["question"]


def test_hugging_face_ids_are_kept(tmp_path):
    assert resolve_model("BAAI/bge-m3", [tmp_path]) == "BAAI/bge-m3"


def test_local_model_folder_is_found_relative_to_the_project(tmp_path):
    (tmp_path / "models" / "bge-m3-egyptian").mkdir(parents=True)
    other = tmp_path / "backend"
    other.mkdir()
    found = resolve_model("models/bge-m3-egyptian", [other, tmp_path])
    assert Path(found) == (tmp_path / "models" / "bge-m3-egyptian").resolve()
    assert resolve_model(str(tmp_path / "models" / "bge-m3-egyptian")) == str(tmp_path / "models" / "bge-m3-egyptian")


def test_missing_local_model_gives_a_clear_error(tmp_path):
    with pytest.raises(ModelNotFoundError, match="egyptian_finetune.ipynb"):
        resolve_model("models/bge-m3-egyptian", [tmp_path])
    with pytest.raises(ModelNotFoundError):
        resolve_model(str(tmp_path / "nowhere"))
