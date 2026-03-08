# src/phase1_ingest/cleaner.py
"""
Cleaner — strips newsgroup-specific noise from raw post text.

Newsgroup posts have a structured header block followed by body text.
The header (From:, Subject:, Lines:, etc.) is metadata noise for semantic
search — it would cause the embedder to cluster by author/server rather
than topic. We strip it.

Additional noise we remove:
  - Quoted reply blocks (lines starting with ">") — these are someone
    else's words, not the document's own content
  - PGP signature blocks — cryptographic boilerplate
  - MIME/UUencoded attachment blocks — binary data encoded as ASCII
  - Extremely short documents — fewer than 50 tokens carry too little
    semantic signal to embed meaningfully
  - Extremely long documents — cap at 512 words to fit embedding context
    window and keep VRAM usage predictable on 4GB GPU
"""

import re
from typing import Optional


# ── Regex patterns compiled once at import ───────────────────────────────────

# Header block: lines before the first blank line that look like "Key: value"
_HEADER_RE = re.compile(r'^[\w\-]+:.*$', re.MULTILINE)

# Quoted reply lines ("> text" or ">> text")
_QUOTE_RE = re.compile(r'^\s*>+.*$', re.MULTILINE)

# PGP blocks
_PGP_RE = re.compile(
    r'-----BEGIN PGP.*?-----END PGP[^-]*-----', re.DOTALL
)

# UUencoded / MIME attachment blocks
_UUENCODE_RE = re.compile(r'^begin \d{3} .*?^end\s*$', re.MULTILINE | re.DOTALL)
_MIME_RE = re.compile(r'Content-Type:.*?(?=\n\n|\Z)', re.DOTALL | re.IGNORECASE)

# Repeated punctuation / ASCII art (3+ same non-word chars in a row)
_ASCII_ART_RE = re.compile(r'[^\w\s]{3,}')

# Multiple blank lines → single blank line
_MULTI_BLANK_RE = re.compile(r'\n{3,}')


def strip_header(text: str) -> str:
    """
    Remove the RFC-2822 style header block at the top of each post.
    Headers end at the first blank line. We drop everything before it.
    """
    # Find the first blank line
    blank_line_match = re.search(r'\n\s*\n', text)
    if blank_line_match:
        return text[blank_line_match.end():]
    return text


def clean_doc(raw_text: str, min_tokens: int = 50, max_tokens: int = 512) -> Optional[str]:
    """
    Full cleaning pipeline for a single newsgroup post.

    Returns cleaned text string, or None if the document should be discarded.

    Args:
        raw_text   : raw file contents
        min_tokens : discard if fewer than this many whitespace-split tokens remain
        max_tokens : truncate to this many tokens (protects embedding context window)
    """
    text = raw_text

    # 1. Strip RFC-2822 header block
    text = strip_header(text)

    # 2. Remove PGP signatures
    text = _PGP_RE.sub('', text)

    # 3. Remove UUencoded / MIME attachment blocks
    text = _UUENCODE_RE.sub('', text)
    text = _MIME_RE.sub('', text)

    # 4. Remove quoted reply lines (">")
    #    Rationale: quoted text is the previous poster's words, not this
    #    document's semantic content. Including it would blur cluster boundaries.
    text = _QUOTE_RE.sub('', text)

    # 5. Remove ASCII art / repeated punctuation
    text = _ASCII_ART_RE.sub(' ', text)

    # 6. Collapse whitespace
    text = _MULTI_BLANK_RE.sub('\n\n', text)
    text = text.strip()

    # 7. Token length filter
    tokens = text.split()
    if len(tokens) < min_tokens:
        return None  # too short — not enough semantic signal

    # 8. Truncate to max_tokens
    #    We keep the first max_tokens words. Newsgroup posts front-load
    #    their main point, so truncation loses mostly noise.
    if len(tokens) > max_tokens:
        text = ' '.join(tokens[:max_tokens])

    return text


def clean_corpus(raw_docs: list[dict]) -> list[dict]:
    """
    Apply clean_doc to an iterable of raw dicts from loader.iter_raw_docs().
    Returns list of cleaned dicts with 'text' field added.
    Discarded documents are dropped from the output.
    """
    cleaned = []
    discarded = 0

    for doc in raw_docs:
        cleaned_text = clean_doc(doc["raw_text"])
        if cleaned_text is None:
            discarded += 1
            continue
        cleaned.append({
            "doc_id"  : doc["doc_id"],
            "category": doc["category"],
            "text"    : cleaned_text,
        })

    total = len(cleaned) + discarded
    print(f"[Cleaner] {len(cleaned):,} kept / {discarded:,} discarded "
          f"({discarded/total*100:.1f}% filtered) from {total:,} total")
    return cleaned