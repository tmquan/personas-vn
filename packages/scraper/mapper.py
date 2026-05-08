"""Convert raw NSO WordPress records into ontology types.

Inputs: the ``records_by_endpoint`` dict returned by
:func:`packages.scraper.nso.scrape_nso`.

Output: a list of :class:`packages.ontology.Dataset` instances ready to
serialise to disk.

We keep the mapper tiny on purpose. It does *no* HTTP and *no* file IO so
it's trivially testable.
"""

from __future__ import annotations

import html
import re
from datetime import datetime
from typing import Any

from packages.common.logging import get_logger
from packages.ontology import Dataset, OntologyRegistry

log = get_logger(__name__)

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _strip_html(value: str | None) -> str:
    """WordPress HTML excerpt -> plain text."""
    if not value:
        return ""
    no_tags = _TAG_RE.sub(" ", value)
    return _WS_RE.sub(" ", html.unescape(no_tags)).strip()


def _parse_date(value: Any) -> datetime | None:
    """WP returns 'YYYY-MM-DDTHH:MM:SS' (no tz). Be liberal in what we accept."""
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _resolve_domain_for_post(
    post: dict[str, Any],
    *,
    category_index: dict[int, dict[str, Any]],
    registry: OntologyRegistry,
) -> str:
    """Pick the best domain id for a post by walking its categories."""
    for cid in post.get("categories", []) or []:
        cat = category_index.get(int(cid))
        if not cat:
            continue
        domain_id = registry.map_nso_category(cat.get("name"), cat.get("slug"))
        if domain_id != "other":
            return domain_id
    return "other"


def map_posts_to_datasets(
    records_by_endpoint: dict[str, list[dict[str, Any]]],
    registry: OntologyRegistry,
) -> list[Dataset]:
    """Project NSO posts onto :class:`Dataset` instances."""
    posts = records_by_endpoint.get("posts", []) or []
    categories = records_by_endpoint.get("categories", []) or []
    tags = records_by_endpoint.get("tags", []) or []

    cat_index: dict[int, dict[str, Any]] = {int(c["id"]): c for c in categories if c.get("id")}
    tag_index: dict[int, str] = {
        int(t["id"]): t.get("name") or t.get("slug") or str(t["id"])
        for t in tags
        if t.get("id")
    }

    datasets: list[Dataset] = []
    for post in posts:
        try:
            pid = post.get("id")
            if pid is None:
                continue
            domain_id = _resolve_domain_for_post(post, category_index=cat_index, registry=registry)
            title_obj = post.get("title") or {}
            title = _strip_html(title_obj.get("rendered") if isinstance(title_obj, dict) else title_obj)
            excerpt_obj = post.get("excerpt") or {}
            summary = _strip_html(
                excerpt_obj.get("rendered") if isinstance(excerpt_obj, dict) else excerpt_obj
            )
            dataset = Dataset(
                id=f"nso:{pid}",
                domain_id=domain_id,
                title=title or f"NSO post {pid}",
                slug=post.get("slug"),
                url=post.get("link"),
                language=post.get("lang") or "vi",
                published_at=_parse_date(post.get("date")),
                summary=summary or None,
                tags=[tag_index[int(tid)] for tid in (post.get("tags") or []) if int(tid) in tag_index],
                raw_categories=[int(c) for c in (post.get("categories") or [])],
                payload={
                    "wp_post_id": pid,
                    "wp_categories": post.get("categories") or [],
                },
            )
            datasets.append(dataset)
        except Exception as exc:
            # One bad post should never sink the whole mapping. Log + skip.
            log.warning("skipping post id=%s: %s", post.get("id"), exc)

    log.info("mapped %d posts -> %d datasets", len(posts), len(datasets))
    return datasets
