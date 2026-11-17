"""How well does the claim verifier (app/services/claims.py) tell supported statements from unsupported ones?

There are no hand-labelled answer claims in the project yet, so this builds a *controlled* test set from the
real passages (the label of every case is known by construction, not judged by a model):

| case | how it is made | correct verdict |
|------|----------------|-----------------|
| supported      | a sentence copied from the passage that answers an evaluation question | supported |
| changed number | the same sentence with one number changed to a value in no passage | unsupported |
| other document | a sentence from a passage of another document, not among the passages | unsupported |
| wrong passage  | one number replaced by a number that only another passage of the context has | partial |
| negated        | the sentence with a negation added ("not" / "لا") | contradicted |

Copied sentences are easier than an LLM's paraphrases, so this measures the checks themselves (numbers,
evidence matching, negation), not overall answer faithfulness — `scripts/evaluate.py --answers` measures
real answers, and its human_review.csv is the sample for a person to check. Only the test questions'
passages are used; the verifier's thresholds were not chosen on them.

    python scripts/evaluate_claims.py --chunks-file chunks.jsonl     # exported by evaluate.py --export-chunks
    python scripts/evaluate_claims.py --chunks-file chunks.jsonl --nli  # + the multilingual NLI model
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
PROJECT = BACKEND.parent

from app.evaluation import runner  # noqa: E402
from app.evaluation.dataset import DEFAULT_SEED, load_questions  # noqa: E402
from app.services import claims, grounding  # noqa: E402
from app.services.retrieval import RetrievedChunk  # noqa: E402
from app.utils.text import detect_language  # noqa: E402

logger = logging.getLogger("claims")
EXPECTED = {"supported": claims.SUPPORTED, "changed number": claims.UNSUPPORTED, "other document": claims.UNSUPPORTED,
            "wrong passage": claims.PARTIAL, "negated": claims.CONTRADICTED}


def chunk(i: int, text: str, meta: dict) -> RetrievedChunk:
    return RetrievedChunk(f"c{i}", text, meta.get("source", ""), int(meta.get("page_start", 0) or 0),
                          int(meta.get("page_end", 0) or 0), 1.0, meta.get("topic", ""), meta.get("section", ""))


def sentences(text: str) -> list[str]:
    body = text.split("\n", 1)[1] if " › " in text.split("\n", 1)[0] and "\n" in text else text
    out = []
    for s in re.split(r"(?<=[.!?؟])\s+|\n", body):
        s = s.strip()
        if len(grounding.content_words(s)) >= 5 and len(s) <= 300:
            out.append(s)
    return out


def negate(sentence: str) -> str:
    if detect_language(sentence) == "ar":
        return re.sub(r"^(\S+)", r"لا \1", sentence, count=1)
    for verb in (" is ", " are ", " should ", " can ", " must ", " will "):
        if verb in sentence:
            return sentence.replace(verb, f"{verb}not ", 1)
    return "It is not true that " + sentence[0].lower() + sentence[1:]


def build_cases(questions, ids, texts, metas, rng: random.Random) -> list[dict]:
    corpus = runner.Corpus(ids, texts, metas)
    by_source = defaultdict(list)
    for i, m in enumerate(metas):
        by_source[m.get("source", "")].append(i)
    cases = []
    for q in questions:
        relevant = sorted(corpus.relevant(q))
        if not relevant:
            continue
        main = relevant[0]
        source = metas[main].get("source", "")
        same_doc = [i for i in by_source[source] if i != main]
        context_idx = [main] + rng.sample(same_doc, min(2, len(same_doc)))
        context = [chunk(k, texts[i], metas[i]) for k, i in enumerate(context_idx)]
        context_numbers = set().union(*(grounding._passage_numbers(c.text) for c in context))
        main_numbers = grounding._passage_numbers(texts[main])
        own = sentences(texts[main])
        if not own:
            continue
        s = rng.choice(own)
        cases.append({"q": q.id, "case": "supported", "claim": f"{s} [1]", "context": context})
        cases.append({"q": q.id, "case": "negated", "claim": f"{negate(s)} [1]", "context": context})
        # a number the sentence states (not a list marker such as "40-"), written plainly as digits
        with_numbers = [x for x in own if any(re.search(rf"(?<![\d.,]){re.escape(n)}(?![\d.,])", grounding.answer_body(x))
                                              for n in grounding.numbers(grounding.answer_body(x)) if n.isdigit())]
        if with_numbers:
            s2 = grounding.answer_body(rng.choice(with_numbers)).strip()
            n = next(n for n in sorted(grounding.numbers(s2)) if n.isdigit() and re.search(rf"(?<![\d.,]){n}(?![\d.,])", s2))
            new = next(str(v) for v in (int(n) * 3 + 7, int(n) * 5 + 11, 977) if str(v) not in context_numbers)
            changed = re.sub(rf"(?<![\d.,]){n}(?![\d.,])", new, s2, count=1)
            if new in grounding.numbers(changed):
                cases.append({"q": q.id, "case": "changed number", "claim": f"{changed} [1]", "context": context})
            others = sorted(x for x in context_numbers - main_numbers if x.isdigit() and len(x) <= 4)
            if others:
                other = rng.choice(others)
                moved = re.sub(rf"(?<![\d.,]){n}(?![\d.,])", other, s2, count=1)
                if other in grounding.numbers(moved) and other not in grounding._passage_numbers(texts[main]):
                    cases.append({"q": q.id, "case": "wrong passage", "claim": f"{moved} [1]", "context": context})
        foreign_sources = [src for src in by_source if src != source]
        foreign = [x for i in by_source[rng.choice(foreign_sources)][:40] for x in sentences(texts[i])
                   if detect_language(x) == detect_language(s)]
        if foreign:
            cases.append({"q": q.id, "case": "other document", "claim": rng.choice(foreign), "context": context})
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chunks-file", type=Path, required=True)
    parser.add_argument("--nli", action="store_true", help="also run the multilingual NLI model")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--split", default="test")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    rng = random.Random(args.seed)
    ids, texts, metas = runner.load_chunk_file(args.chunks_file)
    questions = [q for q in load_questions(PROJECT / "data", seed=args.seed) if q.answerable and q.split == args.split]
    cases = build_cases(questions, ids, texts, metas, rng)
    verifiers = {"lexical + numbers": None}
    if args.nli:
        from app.core.config import get_settings

        settings = get_settings()
        verifiers["+ NLI"] = claims.NLIVerifier(settings.nli_model, allow_download=args.allow_download or settings.nli_allow_download)
    results = {}
    for name, nli in verifiers.items():
        confusion: dict[str, Counter] = defaultdict(Counter)
        for c in cases:
            report = claims.verify(c["claim"], c["context"], "", nli)
            verdict = report.claims[0].verdict if report.claims else "no claim"
            confusion[c["case"]][verdict] += 1
        if nli is not None and nli.failed:
            results[name] = {"status": "NOT RUN", "reason": nli.failed}
            continue
        per_case = {case: {"n": sum(cnt.values()), "correct": round(cnt[EXPECTED[case]] / sum(cnt.values()), 3),
                           "verdicts": dict(cnt)} for case, cnt in confusion.items()}
        flagged = lambda case: sum(v for k, v in confusion[case].items() if k in (claims.UNSUPPORTED, claims.CONTRADICTED, claims.PARTIAL))  # noqa: E731
        bad = ("changed number", "other document", "wrong passage", "negated")
        n_bad = sum(sum(confusion[c].values()) for c in bad)
        results[name] = {
            "status": "OK", "per_case": per_case,
            "unsupported_detected (any non-supported verdict on a bad claim)": round(sum(flagged(c) for c in bad) / n_bad, 3) if n_bad else None,
            "false_alarm_rate (supported claim not marked supported)":
                round(1 - confusion["supported"][claims.SUPPORTED] / max(sum(confusion["supported"].values()), 1), 3),
        }
        logger.info("%s: %s", name, json.dumps({k: v["correct"] for k, v in per_case.items()}))
    out = args.out or PROJECT / "evaluation" / "results" / (datetime.now().strftime("%Y%m%d-%H%M%S") + "-claims")
    out.mkdir(parents=True, exist_ok=True)
    payload = {"date": datetime.now(timezone.utc).isoformat(timespec="seconds"), "split": args.split, "seed": args.seed,
               "cases": dict(Counter(c["case"] for c in cases)), "results": results}
    (out / "claims.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(out / "claims_cases.jsonl", "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps({"q": c["q"], "case": c["case"], "claim": c["claim"],
                                "context": [x.chunk_id for x in c["context"]]}, ensure_ascii=False) + "\n")
    logger.info("Results in %s", out)


if __name__ == "__main__":
    main()
