"""Small, database-backed result pages for the article viewer."""

from __future__ import annotations

from math import ceil
from typing import Any, Dict, Iterable, List, Optional, Sequence


def _provider(store: Any) -> str:
    if hasattr(store, "db") and hasattr(store.db, "articles"):
        return "mongodb"
    if callable(getattr(store, "_connect", None)):
        return "sqlite"
    return "other"


def _fallback_articles(store: Any) -> Iterable[Dict[str, Any]]:
    iterator = getattr(store, "iter_articles", None)
    return iterator() if callable(iterator) else store.list_articles(limit=50000)


def list_sources(store: Any) -> List[str]:
    provider = _provider(store)
    if provider == "mongodb":
        values = store.db.articles.distinct("source")
    elif provider == "sqlite":
        con = store._connect()
        try:
            values = [row[0] for row in con.execute("SELECT DISTINCT source FROM articles")]
        finally:
            con.close()
    else:
        values = [row.get("source") for row in _fallback_articles(store)]
    return sorted({str(value) for value in values if value}, key=str.casefold)


def list_filter_tags(store: Any, categories: Sequence[str]) -> List[Dict[str, Any]]:
    """Read only tags in the selected categories, without embedding fields."""
    categories = list(dict.fromkeys(str(value) for value in categories if value))
    if not categories:
        return []
    provider = _provider(store)
    if provider == "mongodb":
        rows = store.db.tags.find(
            {"category": {"$in": categories}},
            {"_id": 1, "name": 1, "category": 1},
        ).sort("name", 1)
        return [
            {
                "id": int(row["_id"]),
                "name": row.get("name", ""),
                "category": row.get("category", "GENERAL"),
            }
            for row in rows
        ]
    if provider == "sqlite":
        placeholders = ",".join("?" for _ in categories)
        con = store._connect()
        try:
            rows = con.execute(
                "SELECT id, name, category FROM tags "
                f"WHERE category IN ({placeholders}) ORDER BY name",
                categories,
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            con.close()
    return [
        {
            "id": tag.get("id"),
            "name": tag.get("name", ""),
            "category": tag.get("category", "GENERAL"),
        }
        for tag in (store.get_all_tags() or [])
        if tag.get("category") in categories
    ]


def _descendant_ids(store: Any, selected: Sequence[int]) -> set[int]:
    """Keep the existing behavior where a parent tag includes descendants."""
    found = set(selected)
    pending = list(selected)
    reader = getattr(store, "get_tag_relations", None)
    if not callable(reader):
        return found
    while pending:
        relations = reader(pending.pop()) or {}
        for child in relations.get("children", []):
            try:
                child_id = int(child["id"])
            except (KeyError, TypeError, ValueError):
                continue
            if child_id not in found:
                found.add(child_id)
                pending.append(child_id)
    return found


def _page_tags(store: Any, articles: List[Dict[str, Any]]) -> None:
    ids = [str(article["id"]) for article in articles]
    if not ids:
        return
    provider = _provider(store)
    by_article: Dict[str, List[Dict[str, Any]]] = {article_id: [] for article_id in ids}
    if provider == "sqlite":
        placeholders = ",".join("?" for _ in ids)
        con = store._connect()
        try:
            rows = con.execute(
                f"""SELECT at.article_id, t.id, t.name, t.category
                    FROM article_tags at JOIN tags t ON t.id = at.tag_id
                    WHERE at.article_id IN ({placeholders}) ORDER BY t.name""",
                ids,
            ).fetchall()
            for row in rows:
                by_article[str(row["article_id"])].append(
                    {"id": row["id"], "name": row["name"], "category": row["category"]}
                )
        finally:
            con.close()
    elif provider == "mongodb":
        relations = list(
            store.db.article_tags.find(
                {"article_id": {"$in": ids}},
                {"_id": 0, "article_id": 1, "tag_id": 1},
            )
        )
        tag_ids = list({row["tag_id"] for row in relations})
        tags = {
            row["_id"]: {
                "id": row["_id"],
                "name": row.get("name", ""),
                "category": row.get("category", "GENERAL"),
            }
            for row in store.db.tags.find(
                {"_id": {"$in": tag_ids}}, {"_id": 1, "name": 1, "category": 1}
            )
        } if tag_ids else {}
        for row in relations:
            tag = tags.get(row["tag_id"])
            if tag:
                by_article[str(row["article_id"])].append(tag)
    else:
        for article_id in ids:
            by_article[article_id] = [
                {
                    "id": tag.get("id"),
                    "name": tag.get("name", ""),
                    "category": tag.get("category", "GENERAL"),
                }
                for tag in (store.get_article_tags(article_id) or [])
            ]
    for article in articles:
        article["tags"] = sorted(
            by_article[str(article["id"])],
            key=lambda tag: str(tag.get("name") or "").casefold(),
        )


def list_article_page(
    store: Any, *, start_ts: Optional[int], end_ts: Optional[int],
    sources: Sequence[str], categories: Sequence[str], selected_tags: Sequence[int],
    page: int, per_page: int,
) -> Dict[str, Any]:
    """Return a filtered page and exact count; fetch tags only for that page."""
    sources = list(dict.fromkeys(sources))
    categories = list(dict.fromkeys(categories))
    allowed_tags = list_filter_tags(store, categories) if categories else []
    allowed_ids = {int(tag["id"]) for tag in allowed_tags}
    selected_ids = [tag_id for tag_id in selected_tags if tag_id in allowed_ids]
    if selected_ids:
        match_ids = sorted(_descendant_ids(store, selected_ids) & allowed_ids)
    else:
        match_ids = sorted(allowed_ids)
    need_tags = bool(categories)
    impossible = need_tags and not match_ids
    provider = _provider(store)

    if impossible:
        total, rows = 0, []
        page = 1
    elif provider == "sqlite":
        where: List[str] = []
        params: List[Any] = []
        timestamp = "COALESCE(NULLIF(a.published_ts, 0), a.fetched_at, 0)"
        if start_ts is not None:
            where.append(f"{timestamp} >= ?")
            params.append(start_ts)
        if end_ts is not None:
            where.append(f"{timestamp} <= ?")
            params.append(end_ts)
        if sources:
            where.append(f"a.source IN ({','.join('?' for _ in sources)})")
            params.extend(sources)
        if need_tags:
            if selected_ids:
                where.append(
                    "EXISTS (SELECT 1 FROM article_tags at WHERE at.article_id = a.id "
                    f"AND at.tag_id IN ({','.join('?' for _ in match_ids)}))"
                )
                params.extend(match_ids)
            else:
                where.append(
                    "EXISTS (SELECT 1 FROM article_tags at "
                    "JOIN tags t ON t.id = at.tag_id WHERE at.article_id = a.id "
                    f"AND t.category IN ({','.join('?' for _ in categories)}))"
                )
                params.extend(categories)
        clause = " WHERE " + " AND ".join(where) if where else ""
        con = store._connect()
        try:
            total = con.execute("SELECT COUNT(*) FROM articles a" + clause, params).fetchone()[0]
            page = min(page, max(1, ceil(total / per_page)))
            result = con.execute(
                "SELECT a.id, a.title, a.source, a.url, a.published_ts, a.fetched_at "
                "FROM articles a" + clause +
                f" ORDER BY {timestamp} DESC, a.id DESC LIMIT ? OFFSET ?",
                [*params, per_page, (page - 1) * per_page],
            ).fetchall()
            rows = [dict(row) for row in result]
        finally:
            con.close()
    elif provider == "mongodb":
        query: Dict[str, Any] = {}
        bounds: Dict[str, int] = {}
        if start_ts is not None:
            bounds["$gte"] = start_ts
        if end_ts is not None:
            bounds["$lte"] = end_ts
        if bounds:
            query["_mongo_sort_ts"] = bounds
        if sources:
            query["source"] = {"$in": sources}
        if need_tags:
            stages = [
                {"$match": query},
                {"$lookup": {
                    "from": "article_tags", "let": {"article_id": "$_id"},
                    "pipeline": [
                        {"$match": {
                            "tag_id": {"$in": match_ids},
                            "$expr": {"$eq": ["$article_id", "$$article_id"]},
                        }},
                        {"$limit": 1},
                    ],
                    "as": "matching_tags",
                }},
                {"$match": {"matching_tags.0": {"$exists": True}}},
            ]
            counts = list(store.db.articles.aggregate([*stages, {"$count": "total"}]))
            total = counts[0]["total"] if counts else 0
            page = min(page, max(1, ceil(total / per_page)))
            rows = list(store.db.articles.aggregate([
                *stages,
                {"$sort": {"_mongo_sort_ts": -1, "_id": -1}},
                {"$skip": (page - 1) * per_page},
                {"$limit": per_page},
                {"$project": {
                    "_id": 1, "title": 1, "source": 1, "url": 1,
                    "published_ts": 1, "fetched_at": 1,
                }},
            ]))
        else:
            total = store.db.articles.count_documents(query)
            page = min(page, max(1, ceil(total / per_page)))
            rows = list(
                store.db.articles.find(
                    query,
                    {
                        "_id": 1, "title": 1, "source": 1, "url": 1,
                        "published_ts": 1, "fetched_at": 1,
                    },
                ).sort([("_mongo_sort_ts", -1), ("_id", -1)])
                .skip((page - 1) * per_page).limit(per_page)
            )
        rows = [{"id": str(row["_id"]), **{key: row.get(key) for key in (
            "title", "source", "url", "published_ts", "fetched_at"
        )}} for row in rows]
    else:
        matches = []
        for article in _fallback_articles(store):
            ts = int(article.get("published_ts") or article.get("fetched_at") or 0)
            if start_ts is not None and ts < start_ts:
                continue
            if end_ts is not None and ts > end_ts:
                continue
            if sources and article.get("source") not in sources:
                continue
            if need_tags and not any(
                int(tag.get("id") or 0) in match_ids
                for tag in (store.get_article_tags(article["id"]) or [])
            ):
                continue
            matches.append(article)
        matches.sort(
            key=lambda article: (
                int(article.get("published_ts") or article.get("fetched_at") or 0),
                str(article.get("id") or ""),
            ),
            reverse=True,
        )
        total = len(matches)
        page = min(page, max(1, ceil(total / per_page)))
        rows = matches[(page - 1) * per_page:page * per_page]
    articles = [
        {
            "id": str(row["id"]), "title": row.get("title") or "",
            "source": row.get("source") or "", "url": row.get("url") or "",
            "published_ts": int(row.get("published_ts") or 0),
            "fetched_at": int(row.get("fetched_at") or 0),
        }
        for row in rows if row.get("id")
    ]
    _page_tags(store, articles)
    return {
        "articles": articles, "total": total, "page": page,
        "pages": max(1, ceil(total / per_page)),
        "filter_tags": allowed_tags,
        "selected_tags": selected_ids,
    }
