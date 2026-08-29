from __future__ import annotations

import re
from typing import Iterable, Optional


def normalize_title(t: Optional[str]) -> str:
    """Lowercase, replace smart quotes/dashes, collapse non-alphanumerics to spaces."""
    if not t:
        return ""
    t = t.lower()
    # Smart quotes → straight
    t = t.replace("’", "'").replace("‘", "'")
    t = t.replace("“", '"').replace("”", '"')
    # Em/en dashes → hyphen
    t = t.replace("–", "-").replace("—", "-")
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def title_matches(retrieved: Optional[str], excluded: Optional[str]) -> bool:
    """True if `retrieved` looks like the same paper as `excluded`.

    Strategy:
      - Normalize both (case, punctuation, smart quotes / dashes).
      - Exact match → True.
      - Otherwise, if BOTH normalized titles are reasonably long (≥30 chars),
        allow substring match in either direction. Catches:
          * subtitle additions ("Foo: A Bar Approach" vs "Foo")
          * truncations in S2 responses
          * author/venue suffixes
      - Below the length threshold, require exact match — avoids false
        positives on generic short titles like "Attention".
    """
    r = normalize_title(retrieved)
    e = normalize_title(excluded)
    if not r or not e:
        return False
    if r == e:
        return True
    if min(len(r), len(e)) >= 30:
        return (e in r) or (r in e)
    return False


def is_excluded(retrieved_title: Optional[str], exclude_titles: Iterable[str]) -> bool:
    """True if `retrieved_title` matches any title in `exclude_titles`."""
    if not retrieved_title or not exclude_titles:
        return False
    for ex in exclude_titles:
        if title_matches(retrieved_title, ex):
            return True
    return False
