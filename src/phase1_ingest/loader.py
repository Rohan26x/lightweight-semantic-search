# src/phase1_ingest/loader.py
"""
Loader — walks the 20_newsgroups directory tree and yields raw documents.

Directory structure:
  20_newsgroups/
    <category>/        e.g. alt.atheism, sci.space, talk.politics.guns
      <doc_id>         raw text file (no extension)

Each file is one newsgroup post. We extract:
  - doc_id   : "<category>/<filename>"  (unique across corpus)
  - category : folder name (the ground-truth label we keep as metadata)
  - raw_text : full file contents
"""

import os
from pathlib import Path
from typing import Iterator
from src.config import RAW_DATA_DIR


def iter_raw_docs(data_dir: Path = RAW_DATA_DIR) -> Iterator[dict]:
    """
    Yields dicts: {doc_id, category, raw_text}
    Skips unreadable files silently (a handful have encoding issues).
    """
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Raw data not found at {data_dir}.\n"
            "Extract the tar.gz into data/raw/ so the path "
            "data/raw/20_newsgroups/<category>/<file> exists."
        )

    categories = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    print(f"[Loader] Found {len(categories)} categories in {data_dir}")

    total = 0
    for cat_dir in categories:
        for fpath in sorted(cat_dir.iterdir()):
            if not fpath.is_file():
                continue
            # Try UTF-8 first, fall back to latin-1
            # (some posts contain non-UTF-8 byte sequences)
            for enc in ("utf-8", "latin-1"):
                try:
                    raw_text = fpath.read_text(encoding=enc)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                continue  # skip if both encodings fail

            yield {
                "doc_id"  : f"{cat_dir.name}/{fpath.name}",
                "category": cat_dir.name,
                "raw_text": raw_text,
            }
            total += 1

    print(f"[Loader] Loaded {total:,} raw documents.")


def count_docs(data_dir: Path = RAW_DATA_DIR) -> int:
    return sum(
        1 for cat in data_dir.iterdir() if cat.is_dir()
        for f in cat.iterdir() if f.is_file()
    )