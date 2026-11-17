"""Switch SAQYA's optional AI/ML layers on or off in backend/.env, safely.

    python scripts/enable_ai_layers.py               # the settings the evaluation chose (see below)
    python scripts/enable_ai_layers.py --confidence  # also confidence (after: python scripts/evaluate.py --translate --calibrate)
    python scripts/enable_ai_layers.py --off         # back to the original pipeline

What it does:
- copies backend/.env to backend/.env.backup-<date-time> first (git-ignored: it holds your keys);
- changes only the lines of the settings below (adding them at the end if missing); every other line, including
  your Gemini key, stays exactly as it was;
- prints only the names of the settings it changed, never any value from the file.
Restart the backend afterwards.

The settings come from the evaluation on your PC (docs/ai-ml-engineering.md, section D): query expansion + keyword
weight 0.5 + no reserved keyword slot + reranker raised test MRR@10 from 0.792 to 0.920.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]

CHOSEN = {
    "QUERY_EXPANSION_ENABLED": "true",
    "KEYWORD_WEIGHT": "0.5",
    "KEYWORD_SEAT": "false",
    "RERANKER_ENABLED": "true",
    "QUERY_CLASSIFICATION_ENABLED": "true",
    "CLAIM_VERIFICATION_ENABLED": "true",
}
ORIGINAL = {
    "QUERY_EXPANSION_ENABLED": "false",
    "KEYWORD_WEIGHT": "1.0",
    "KEYWORD_SEAT": "true",
    "RERANKER_ENABLED": "false",
    "QUERY_CLASSIFICATION_ENABLED": "false",
    "CLAIM_VERIFICATION_ENABLED": "false",
    "CONFIDENCE_CALIBRATION_ENABLED": "false",
}
_SETTING = re.compile(r"^\s*([A-Z_][A-Z0-9_]*)\s*=")


def update(path: Path, wanted: dict[str, str]) -> tuple[list[str], list[str]]:
    raw = path.read_bytes()
    bom = raw.startswith(b"\xef\xbb\xbf")
    text = raw[3:].decode("utf-8") if bom else raw.decode("utf-8")
    newline = "\r\n" if "\r\n" in text else "\n"
    changed, seen, out = [], set(), []
    for line in text.split(newline):
        m = _SETTING.match(line)
        if m and m.group(1) in wanted:
            name = m.group(1)
            if name in seen:
                continue   # a duplicate would make the setting ambiguous
            seen.add(name)
            new = f"{name}={wanted[name]}"
            if line.strip() != new:
                changed.append(name)
            out.append(new)
        else:
            out.append(line)
    added = [k for k in wanted if k not in seen]
    if added:
        while out and not out[-1].strip():
            out.pop()
        out += ["", f"# ---- Optional AI/ML layers (scripts/enable_ai_layers.py, {datetime.now():%Y-%m-%d})"]
        out += [f"{k}={wanted[k]}" for k in added] + [""]
    data = newline.join(out).encode("utf-8")
    path.write_bytes((b"\xef\xbb\xbf" if bom else b"") + data)
    return changed, added


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--confidence", action="store_true", help="also switch on CONFIDENCE_CALIBRATION_ENABLED")
    parser.add_argument("--off", action="store_true", help="switch every optional layer off (original pipeline)")
    args = parser.parse_args()
    env = BACKEND / ".env"
    if not env.exists():
        sys.exit("backend/.env not found. Copy backend/.env.example to backend/.env first.")
    wanted = dict(ORIGINAL) if args.off else dict(CHOSEN)
    if args.confidence and not args.off:
        if not (BACKEND / "data" / "confidence_calibration.json").exists():
            print("Note: no calibration file yet (python scripts/evaluate.py --translate --calibrate); confidence will "
                  "only ask back for a missing crop and add cautions, without probabilities.")
        wanted["CONFIDENCE_CALIBRATION_ENABLED"] = "true"
    stamp = f"{datetime.now():%Y%m%d-%H%M%S}"
    backup, n = env.with_name(f".env.backup-{stamp}"), 1
    while backup.exists():   # never overwrite an earlier backup
        backup, n = env.with_name(f".env.backup-{stamp}-{n}"), n + 1
    shutil.copy2(env, backup)
    changed, added = update(env, wanted)
    print(f"Backup of your previous settings: {backup.name}")
    if not changed and not added:
        print("Nothing to change: the settings were already like this.")
    for name in changed:
        print(f"  changed: {name} = {wanted[name]}")
    for name in added:
        print(f"  added:   {name} = {wanted[name]}")
    print("Done. Restart the backend, then open http://localhost:8000/health and look at 'ai_layers'.")


if __name__ == "__main__":
    main()
