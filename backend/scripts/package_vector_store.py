"""Pack the vector store into one zip a team member can download (it is too big for git: GitHub refuses files
over 100 MB, and it is built from PDFs that are not in the repo).

    python scripts/package_vector_store.py            # -> saqya-vector-store.zip in the project folder

Only the active collection's files are packed: chroma.sqlite3, the index folder(s) of the collection named in
rag_config.json, rag_config.json itself and, if present, data/confidence_calibration.json. Old index folders
left by earlier builds are skipped. Stop the backend first so the database is not being written.

The team member unzips it into backend/ (it recreates backend/data/vector_store/); docs/ONBOARDING.md, step 4.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import zipfile
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
PROJECT = BACKEND.parent


def active_segments(store: Path, collection: str) -> set[str] | None:
    """Folder names of the collection's index segments, from Chroma's own database (None if unreadable)."""
    try:
        con = sqlite3.connect(f"file:{store / 'chroma.sqlite3'}?mode=ro", uri=True)
        rows = con.execute("SELECT s.id FROM segments s JOIN collections c ON s.collection = c.id WHERE c.name = ?",
                           (collection,)).fetchall()
        con.close()
        return {r[0] for r in rows}
    except Exception as exc:  # noqa: BLE001
        print(f"Could not read the segment list ({exc}); packing every index folder.")
        return None


def main() -> None:
    sys.path.insert(0, str(BACKEND))
    from app.core.config import get_settings

    settings = get_settings()
    store = settings.vector_store_path
    config_file = store / "rag_config.json"
    if not (store / "chroma.sqlite3").exists() or not config_file.exists():
        sys.exit(f"No vector store at {store}.")
    config = json.loads(config_file.read_text(encoding="utf-8"))
    collection = settings.collection_name or config["collection_name"]
    keep = active_segments(store, collection)
    folders = [d for d in store.iterdir() if d.is_dir() and (keep is None or d.name in keep)]
    skipped = [d.name for d in store.iterdir() if d.is_dir() and d not in folders]
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT / "saqya-vector-store.zip"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        # fixed paths inside the zip, so it unpacks into backend/ wherever the store lives on this machine
        for f in [store / "chroma.sqlite3", config_file] + [p for d in folders for p in d.rglob("*") if p.is_file()]:
            z.write(f, "data/vector_store/" + f.relative_to(store).as_posix())
        calibration = settings.confidence_calibration_path
        if calibration.exists():
            z.write(calibration, "data/confidence_calibration.json")
    print(f"Wrote {target} ({target.stat().st_size / 2**20:.0f} MB): collection '{collection}', "
          f"{config.get('num_chunks', '?')} passages, embedding model {config.get('embedding_model')}.")
    if skipped:
        print(f"Skipped {len(skipped)} old index folder(s) not used by the collection.")
    print("Share it (e.g. attach it to a GitHub release of a private repo); the team member unzips it into backend/.")


if __name__ == "__main__":
    main()
