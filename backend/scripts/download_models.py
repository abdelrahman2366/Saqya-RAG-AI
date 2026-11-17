"""Download an optional model into models/ — only when you run this (the app never downloads at startup).

    python scripts/download_models.py reranker        # BAAI/bge-reranker-v2-m3 (~2.3 GB, Apache-2.0)
    python scripts/download_models.py reranker-jina   # jinaai/jina-reranker-v2-base-multilingual (~1.1 GB, CC BY-NC 4.0)
    python scripts/download_models.py nli             # MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 (~0.6 GB)
    python scripts/download_models.py <any Hugging Face id>

The app finds a model in models/<last part of its name> automatically (reranker.local_model_path), so after
downloading set e.g. RERANKER_ENABLED=true; no download flag is needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
ALIASES = {
    "reranker": "BAAI/bge-reranker-v2-m3",
    "reranker-jina": "jinaai/jina-reranker-v2-base-multilingual",
    "nli": "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
    "embedding": "BAAI/bge-m3",
}


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    name = ALIASES.get(sys.argv[1], sys.argv[1])
    from huggingface_hub import snapshot_download

    target = PROJECT / "models" / name.split("/")[-1]
    if target.exists() and any(target.glob("config.json")):
        print(f"Already there: {target}")
        return
    print(f"Downloading {name} to {target} ...")
    snapshot_download(name, local_dir=str(target), allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model",
                                                                  "*.py", "tokenizer*", "sentencepiece*", "*.spm"])
    print(f"Done: {target}")


if __name__ == "__main__":
    main()
