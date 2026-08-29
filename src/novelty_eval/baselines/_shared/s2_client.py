"""Async Semantic Scholar adapter that vendored baseline code can import as a
drop-in for `papers_from_search_api(query)`.

Single source of truth for S2 calls is `src/semantic_scholar.SEMANTIC_API` (the
same client used by `retrieve_candidates.py` and the rest of the codebase).
This module just (a) goes async via asyncio.to_thread, (b) reshapes the
keyed-dict result into S2's `{"data": [...]}` shape that AI Scientist / CoI-Agent
/ Scideator vendored code expects, and (c) requests the wider field set those
baselines read.

Baselines should `from ..._shared.s2_client import papers_from_search_api` so
their upstream logic stays byte-identical.
"""
from __future__ import annotations

import asyncio
import contextvars
import dataclasses
import os
import threading
from typing import Any, Dict, List, Optional

import aiohttp

from semantic_scholar import SEMANTIC_API, S2_SEARCH_LIMITER, sanitize_s2_query

from .leakage_filter import is_excluded


# Single source of truth for the S2 key — matches `SEMANTIC_API`'s lookup chain.
# Env vars win, then fall back to secrets.toml via `SEMANTIC_API.api_key`. The
# secrets.toml fallback used to be missing, which made every aiohttp path here
# go out unauthenticated and get rate-limited far harder than the sync path.
def _s2_key() -> Optional[str]:
    return (
        os.getenv("S2_API_KEY")
        or os.getenv("SEMENTIC_SEARCH_API_KEY")
        or SEMANTIC_API.api_key
        or None
    )


def _s2_headers() -> Dict[str, str]:
    key = _s2_key()
    return {"x-api-key": key} if key else {}


_S2_BASE = "https://api.semanticscholar.org"


# Per-instance leakage filter. Baselines that thread `exclude_titles` explicitly
# (AI-Scientist) keep doing so; those whose vendored code has no route for it
# (Scideator) inherit it from here, set by the adapter around each instance.
#
# A ContextVar rather than an env var because this genuinely varies per instance
# and instances run concurrently. Propagation is verified for every path S2 calls take:
# directly in a coroutine, through `asyncio.to_thread`, and through a nested
# `asyncio.run` under nest_asyncio. It does NOT survive `loop.run_in_executor`,
# which no S2 path uses.
_EXCLUDE_TITLES: contextvars.ContextVar[tuple] = contextvars.ContextVar(
    "s2_exclude_titles", default=(),
)


def set_exclude_titles(titles: Optional[List[str]]):
    """Install the per-instance exclusion list; returns a token for `reset_exclude_titles`."""
    return _EXCLUDE_TITLES.set(tuple(titles or ()))


def reset_exclude_titles(token) -> None:
    _EXCLUDE_TITLES.reset(token)


def _resolve_exclude_titles(exclude_titles: Optional[List[str]]) -> List[str]:
    """An explicit argument wins; otherwise fall back to the contextvar."""
    if exclude_titles is not None:
        return list(exclude_titles)
    return list(_EXCLUDE_TITLES.get())


@dataclasses.dataclass
class RetrievalStats:
    """What S2 was asked for and what came back, for one instance.

    A paper is **usable** when it carries a top-level `corpusId`: the vendored
    retrieval guards discard everything else before ranking, so those entries
    never reach the judge even though S2 returned them.
    """

    search_calls: int = 0
    papers_returned: int = 0
    excluded_by_leakage: int = 0
    entries_without_corpus_id: int = 0  # discarded by downstream callers
    ids_requested: int = 0
    ids_unresolved: int = 0
    chunks_failed: int = 0
    _lock: threading.Lock = dataclasses.field(
        default_factory=threading.Lock, repr=False, compare=False,
    )

    def add(self, **counts: int) -> None:
        with self._lock:
            for key, value in counts.items():
                setattr(self, key, getattr(self, key) + value)

    def as_dict(self) -> Dict[str, int]:
        with self._lock:
            d = {
                f.name: getattr(self, f.name)
                for f in dataclasses.fields(self)
                if not f.name.startswith("_")
            }
        # Unintended loss only. Papers dropped by the leakage filter are the
        # filter working as designed, so counting them as attrition would make
        # every instance look degraded; they stay visible as their own counter.
        usable = d["papers_returned"] - d["entries_without_corpus_id"]
        asked = d["papers_returned"] + d["ids_unresolved"]
        d["usable_papers"] = usable
        d["retained_fraction"] = round(usable / asked, 4) if asked else 1.0
        return d


_RETRIEVAL_STATS: contextvars.ContextVar[Optional[RetrievalStats]] = contextvars.ContextVar(
    "s2_retrieval_stats", default=None,
)


def set_retrieval_stats(stats: Optional[RetrievalStats]):
    """Install a per-instance collector; returns a token for `reset_retrieval_stats`."""
    return _RETRIEVAL_STATS.set(stats)


def reset_retrieval_stats(token) -> None:
    _RETRIEVAL_STATS.reset(token)


def _record(**counts: int) -> None:
    stats = _RETRIEVAL_STATS.get()
    if stats is not None:
        stats.add(**counts)


def _count_missing_corpus_id(papers: List[Dict]) -> int:
    return sum(1 for p in papers if not (isinstance(p, dict) and p.get("corpusId")))


def _resolve_cutoff(cutoff_date: Optional[str]) -> Optional[str]:
    """Normalize a cutoff into an S2 `publicationDateOrYear` value.

    Defaults from `S2_CUTOFF_DATE` env var (set by `run_baselines.py`).
    Accepts "2025-06-01" / "2025-06" / "2025" — anything that looks like a
    date or year. Returns the S2 `:YYYY-MM-DD` (before this date) syntax for
    date-shaped inputs, or `-YYYY` (up to and including year) for year-only.
    """
    cd = cutoff_date if cutoff_date is not None else os.getenv("S2_CUTOFF_DATE", "")
    if not cd:
        return None
    cd = cd.strip()
    # If user already passed an explicit S2 range syntax, respect it.
    if cd.startswith(":") or cd.endswith(":") or cd.startswith("-") or cd.endswith("-") or ":" in cd:
        return cd
    # YYYY-MM-DD or YYYY-MM → use date-range "before"
    if "-" in cd:
        return f":{cd}"
    # Bare year → use S2 year-range "up to and including"
    return f"-{cd}"

# Fields the vendored baselines read from each paper. `corpusId` is required
# at the top level — Scideator's keyword/snippet filters use `p.get("corpusId")`
# directly, and S2 only returns `CorpusId` nested in `externalIds` unless we
# ask for the top-level `corpusId` field explicitly.
_DEFAULT_FIELDS = (
    "title,authors,abstract,paperId,corpusId,year,venue,citationCount,publicationDate,externalIds"
)


def _normalise_authors(authors_raw: Any) -> str:
    """S2 returns authors as `[{"name": ..., "authorId": ...}, ...]`. Some
    baselines expect a comma-joined string (AI Scientist's paper-strings format).
    Keep both: a list under `authors_list` and a string under `authors`.
    """
    if isinstance(authors_raw, list):
        names = [
            (a.get("name", "") if isinstance(a, dict) else str(a))
            for a in authors_raw
        ]
        return ", ".join(filter(None, names))
    if isinstance(authors_raw, str):
        return authors_raw
    return ""


def _to_s2_shape(papers_dict: Dict[str, Dict]) -> Dict[str, Any]:
    """Reshape `{paperId: paper}` into `{"data": [paper, ...], "total": N}`."""
    data: List[Dict[str, Any]] = []
    for pid, p in papers_dict.items():
        out = dict(p)
        out.setdefault("paperId", pid)
        # Scideator's filters require top-level `corpusId`. If the field set
        # didn't include it explicitly, lift it from externalIds.CorpusId.
        if not out.get("corpusId"):
            ext = p.get("externalIds") or {}
            cid = ext.get("CorpusId") if isinstance(ext, dict) else None
            if cid is not None:
                out["corpusId"] = cid
        out["authors_list"] = p.get("authors", [])
        out["authors"] = _normalise_authors(p.get("authors", []))
        out["title"] = p.get("title", "") or ""
        out["abstract"] = p.get("abstract", "") or ""
        out["year"] = p.get("year", "") or ""
        out["venue"] = p.get("venue", "") or ""
        out["citationCount"] = p.get("citationCount", 0) or 0
        data.append(out)
    return {"data": data, "total": len(data)}


async def papers_from_search_api(
    query: str,
    *,
    limit: int = 10,
    fields: Optional[str] = None,
    max_retries_per_page: int = 3,
    search_type: str = "keyword",  # accepted but currently always keyword via SEMANTIC_API
    cutoff_date: Optional[str] = None,  # defaults to S2_CUTOFF_DATE env var
    exclude_titles: Optional[List[str]] = None,  # drop retrieved papers matching any title
) -> Dict[str, Any]:
    """Async wrapper around `SEMANTIC_API.search_papers`.

    Returns `{"data": [paper_dict, ...], "total": N}` to match the upstream
    S2 endpoint shape that vendored baseline code consumes.

    `search_type` is accepted for API parity with the original
    `noveltychecker.utils.s2_api.papers_from_search_api` signature, but
    is currently a no-op — keyword search is the only mode used by these
    baselines.

    `cutoff_date` (optional): "YYYY-MM-DD" / "YYYY-MM" / "YYYY". Filters at S2
    via `publicationDateOrYear`. Defaults to env var `S2_CUTOFF_DATE` so every
    baseline can inherit a global cutoff without threading kwargs everywhere.
    """
    if not query:
        return {"data": [], "total": 0}

    fields_to_use = fields or _DEFAULT_FIELDS
    keyed = await asyncio.to_thread(
        SEMANTIC_API.search_papers,
        query,
        max_results=limit,
        max_retries_per_page=max_retries_per_page,
        fields=fields_to_use,
        publication_date_or_year=_resolve_cutoff(cutoff_date),
    )
    result = _to_s2_shape(keyed or {})
    _excl = _resolve_exclude_titles(exclude_titles)
    if _excl:
        before = len(result["data"])
        result["data"] = [
            p for p in result["data"]
            if not is_excluded(p.get("title", ""), _excl)
        ]
        result["total"] = len(result["data"])
        result["_excluded_count"] = before - len(result["data"])
    _record(
        search_calls=1,
        papers_returned=len(result["data"]),
        excluded_by_leakage=result.get("_excluded_count", 0),
        entries_without_corpus_id=_count_missing_corpus_id(result["data"]),
    )
    return result


async def papers_from_search_api_with_snippet_mode(
    query: str,
    *,
    search_type: str = "keyword",
    limit: int = 100,
    start_year: str = "",
    end_year: str = "",
    cutoff_date: Optional[str] = None,  # defaults to S2_CUTOFF_DATE env var
    exclude_titles: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Scideator's vendored `papers_from_search_api` accepts `search_type` =
    'keyword' or 'snippet'. Snippet mode hits the S2 snippet endpoint and
    returns its raw response (a list of {paper, snippet, score} entries) under
    `data`. Keyword mode delegates to the keyword search above.

    `cutoff_date`, `exclude_titles`: same semantics as in `papers_from_search_api`.
    """
    if search_type == "keyword":
        return await papers_from_search_api(
            query, limit=limit, cutoff_date=cutoff_date, exclude_titles=exclude_titles,
        )

    if search_type != "snippet":
        return {"data": [], "total": 0}

    # Free-form idea text overflows the snippet endpoint's query length and
    # 400s on unusual punctuation. Same treatment as `SEMANTIC_API.search_papers`.
    q = sanitize_s2_query(query)
    if not q:
        return {"data": [], "total": 0}
    params: Dict[str, Any] = {"query": q, "limit": limit}
    year_param = ""
    if start_year and end_year:
        year_param = f"{start_year}-{end_year}"
    elif end_year:
        year_param = f"-{end_year}"
    elif start_year:
        year_param = f"{start_year}-"
    if year_param:
        params["year"] = year_param
    # Date-level cutoff via publicationDateOrYear (takes precedence when set).
    pdoy = _resolve_cutoff(cutoff_date)
    if pdoy:
        params["publicationDateOrYear"] = pdoy

    # Snippet endpoint is much more aggressively rate-limited than
    # /paper/search — 1 RPS isn't enough. Use longer backoff + more retries.
    max_attempts = 5
    async with aiohttp.ClientSession() as session:
        for attempt in range(max_attempts):
            try:
                await S2_SEARCH_LIMITER.wait_async()
                async with session.get(
                    f"{_S2_BASE}/graph/v1/snippet/search/",
                    params=params,
                    headers=_s2_headers(),
                    timeout=60,
                ) as resp:
                    if resp.status == 200:
                        body = await resp.json()
                        _excl = _resolve_exclude_titles(exclude_titles)
                        if _excl and isinstance(body, dict) and body.get("data"):
                            before = len(body["data"])
                            body["data"] = [
                                item for item in body["data"]
                                if not is_excluded(
                                    (item.get("paper") or {}).get("title", "") if isinstance(item, dict) else "",
                                    _excl,
                                )
                            ]
                            body["_excluded_count"] = before - len(body["data"])
                        _entries = body.get("data") or [] if isinstance(body, dict) else []
                        _record(
                            search_calls=1,
                            papers_returned=len(_entries),
                            excluded_by_leakage=(body.get("_excluded_count", 0)
                                                 if isinstance(body, dict) else 0),
                            entries_without_corpus_id=sum(
                                1 for it in _entries
                                if not (isinstance(it, dict)
                                        and (it.get("paper") or {}).get("corpusId"))
                            ),
                        )
                        return body
                    if resp.status in (429, 500, 502, 503, 504):
                        retry_after = resp.headers.get("Retry-After")
                        try:
                            wait_s = float(retry_after) if retry_after else max(5.0 * (2 ** attempt), 5.0)
                        except ValueError:
                            wait_s = max(5.0 * (2 ** attempt), 5.0)
                        print(
                            f"s2_client snippet-search {resp.status} attempt "
                            f"{attempt + 1}/{max_attempts}; sleeping {wait_s:.1f}s"
                        )
                        await asyncio.sleep(wait_s)
                        continue
                    # Surface non-retriable failures (400 etc.) instead of
                    # silently returning empty — this hid a query-sanitization
                    # bug for a long time.
                    body_text = (await resp.text())[:200]
                    print(
                        f"s2_client snippet-search {resp.status} for query "
                        f"{q!r}: {body_text}"
                    )
                    return {"data": [], "total": 0}
            except Exception as e:
                print(f"s2_client snippet-search exception attempt {attempt + 1}/{max_attempts}: {e!r}")
                await asyncio.sleep(max(5.0 * (2 ** attempt), 5.0))
    print(f"s2_client snippet-search gave up after {max_attempts} attempts for query {q!r}")
    return {"data": [], "total": 0}


async def get_paper_data(
    id_: Any,
    id_type: str = "corpus_id",
    batch_wise: bool = False,
    *,
    fields: str = (
        "corpusId,paperId,url,title,year,abstract,authors.name,"
        "fieldsOfStudy,citationCount,venue,publicationDate"
    ),
) -> Any:
    """Async fetch of full paper metadata by paperId or corpusId.

    Single mode (batch_wise=False) → GET /paper/CorpusId:<id> or /paper/<paperId>
    Batch mode (batch_wise=True)   → POST /paper/batch with {"ids": [...]}
    """
    headers = _s2_headers()

    if not batch_wise:
        if id_type == "paper_id":
            url = f"{_S2_BASE}/graph/v1/paper/{id_}"
        else:
            url = f"{_S2_BASE}/graph/v1/paper/CorpusId:{id_}"
        async with aiohttp.ClientSession() as session:
            for attempt in range(3):
                try:
                    await S2_SEARCH_LIMITER.wait_async()
                    async with session.get(
                        url, params={"fields": fields}, headers=headers, timeout=60
                    ) as resp:
                        if resp.status == 200:
                            _one = await resp.json()
                            _record(search_calls=1,
                                    papers_returned=1 if _one else 0,
                                    ids_requested=1,
                                    ids_unresolved=0 if _one else 1)
                            return _one
                        if resp.status in (429, 500, 502, 503, 504):
                            await asyncio.sleep(2 ** attempt)
                            continue
                        return None
                except Exception:
                    await asyncio.sleep(2 ** attempt)
        return None

    # batch
    url = f"{_S2_BASE}/graph/v1/paper/batch"
    async with aiohttp.ClientSession() as session:
        for attempt in range(3):
            try:
                await S2_SEARCH_LIMITER.wait_async()
                async with session.post(
                    url,
                    params={"fields": fields},
                    headers=headers,
                    json={"ids": id_},
                    timeout=120,
                ) as resp:
                    if resp.status == 200:
                        _batch = await resp.json()
                        _record(
                            search_calls=1,
                            ids_requested=len(id_),
                            ids_unresolved=sum(1 for r in (_batch or []) if not r),
                            papers_returned=sum(1 for r in (_batch or []) if r),
                            entries_without_corpus_id=_count_missing_corpus_id(
                                [r for r in (_batch or []) if r]
                            ),
                        )
                        return _batch
                    if resp.status in (429, 500, 502, 503, 504):
                        await asyncio.sleep(2 ** attempt)
                        continue
                    return []
            except Exception:
                await asyncio.sleep(2 ** attempt)
    _record(chunks_failed=1, ids_requested=len(id_) if isinstance(id_, list) else 1)
    return []


def _filter_papers_by_cutoff(papers: list[dict], cutoff_date: Optional[str]) -> list[dict]:
    """Post-filter a list of S2 paper dicts by `publicationDate` / `year`.
    Used for endpoints that don't support `publicationDateOrYear` (e.g. the
    recommendation API). Best-effort — papers missing both fields are kept.
    """
    cd = cutoff_date if cutoff_date is not None else os.getenv("S2_CUTOFF_DATE", "")
    if not cd:
        return papers
    cd = cd.strip().lstrip(":-").rstrip(":-")
    if not cd:
        return papers
    # Parse cutoff into (year, month, day) for comparison
    try:
        parts = cd.split("-")
        cy = int(parts[0])
        cm = int(parts[1]) if len(parts) > 1 else 12
        cdy = int(parts[2]) if len(parts) > 2 else 31
    except (ValueError, IndexError):
        return papers
    out = []
    for p in papers:
        pub = (p.get("publicationDate") or "").strip()
        kept = True
        if pub:
            try:
                pp = pub.split("-")
                py, pm, pdy = int(pp[0]), int(pp[1]) if len(pp) > 1 else 1, int(pp[2]) if len(pp) > 2 else 1
                kept = (py, pm, pdy) <= (cy, cm, cdy)
            except (ValueError, IndexError):
                kept = True
        else:
            y = p.get("year")
            if y is not None:
                try:
                    kept = int(y) <= cy
                except (ValueError, TypeError):
                    kept = True
        if kept:
            out.append(p)
    return out


async def papers_from_recommendation_api_allCs(
    corpus_id: Optional[str] = None, limit: int = 100,
    cutoff_date: Optional[str] = None,
    exclude_titles: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """S2 recommendation API — pool 'all-cs' (broad CS recommendation)."""
    if not corpus_id:
        return {"recommendedPapers": []}
    url = (
        f"{_S2_BASE}/recommendations/v1/papers/forpaper/CorpusId:{corpus_id}"
        f"?fields=corpusId,paperId,title,abstract,url,venue,publicationDate,"
        f"fieldsOfStudy,authors&limit={limit}&from=all-cs"
    )
    async with aiohttp.ClientSession() as session:
        for attempt in range(3):
            try:
                await S2_SEARCH_LIMITER.wait_async()
                async with session.get(url, headers=_s2_headers(), timeout=60) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        recs = data.get("recommendedPapers") or []
                        recs = _filter_papers_by_cutoff(recs, cutoff_date)
                        _excl = _resolve_exclude_titles(exclude_titles)
                        if _excl:
                            recs = [p for p in recs if not is_excluded(p.get("title", ""), _excl)]
                        _record(search_calls=1, papers_returned=len(recs),
                                entries_without_corpus_id=_count_missing_corpus_id(recs))
                        return {"recommendedPapers": recs}
                    if resp.status in (429, 500, 502, 503, 504):
                        await asyncio.sleep(2 ** attempt)
                        continue
                    return {"recommendedPapers": []}
            except Exception:
                await asyncio.sleep(2 ** attempt)
    return {"recommendedPapers": []}


async def papers_from_recommendation_api_recent(
    corpus_id: Optional[str] = None, limit: int = 100,
    cutoff_date: Optional[str] = None,
    exclude_titles: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """S2 recommendation API — pool 'recent' (recent papers only)."""
    if not corpus_id:
        return {"recommendedPapers": []}
    url = (
        f"{_S2_BASE}/recommendations/v1/papers/forpaper/CorpusId:{corpus_id}"
        f"?fields=corpusId,paperId,title,abstract,url,venue,publicationDate,"
        f"fieldsOfStudy,authors&limit={limit}&from=recent"
    )
    async with aiohttp.ClientSession() as session:
        for attempt in range(3):
            try:
                await S2_SEARCH_LIMITER.wait_async()
                async with session.get(url, headers=_s2_headers(), timeout=60) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        recs = data.get("recommendedPapers") or []
                        recs = _filter_papers_by_cutoff(recs, cutoff_date)
                        _excl = _resolve_exclude_titles(exclude_titles)
                        if _excl:
                            recs = [p for p in recs if not is_excluded(p.get("title", ""), _excl)]
                        _record(search_calls=1, papers_returned=len(recs),
                                entries_without_corpus_id=_count_missing_corpus_id(recs))
                        return {"recommendedPapers": recs}
                    if resp.status in (429, 500, 502, 503, 504):
                        await asyncio.sleep(2 ** attempt)
                        continue
                    return {"recommendedPapers": []}
            except Exception:
                await asyncio.sleep(2 ** attempt)
    return {"recommendedPapers": []}
