# LICENSE HEADER MANAGED BY add-license-header
#
# BSD 3-Clause License
#
# Copyright (c) 2026, Martin Vesterlund
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Automatic tagging for persisted summary documents."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

from feedsummary_core.summarizer.tagging import TagManager
from feedsummary_core.tagging_rules import CVE_PATTERN, is_cve_tag


DEFAULT_SUMMARY_MAX_TAGS = 20


def _summary_text(summary_doc: Dict[str, Any]) -> str:
    """Return the same published text that the summary viewer prioritizes."""
    return str(
        summary_doc.get("proofread_revised_summary")
        or summary_doc.get("proofread_published_summary")
        or summary_doc.get("summary")
        or ""
    ).strip()


def _stored_tag(tag: Dict[str, Any]) -> Dict[str, Any]:
    """Keep the portable tag fields needed to render and resolve the tag later."""
    stored = {
        "id": tag.get("id"),
        "name": str(tag.get("name") or "").strip(),
        "category": str(tag.get("category") or "GENERAL").strip() or "GENERAL",
    }
    reasoning = str(tag.get("reasoning") or "").strip()
    if reasoning:
        stored["reasoning"] = reasoning
    return stored


def _summary_max_tags(config: Dict[str, Any], explicit: int | None) -> int:
    if explicit is not None:
        return max(1, int(explicit))
    tagging = config.get("tagging") if isinstance(config, dict) else None
    tagging = tagging if isinstance(tagging, dict) else {}
    try:
        return max(1, int(tagging.get("summary_max_tags", DEFAULT_SUMMARY_MAX_TAGS)))
    except (TypeError, ValueError):
        return DEFAULT_SUMMARY_MAX_TAGS


def _summary_include_cve_tags(config: Dict[str, Any]) -> bool:
    tagging = config.get("tagging") if isinstance(config, dict) else None
    tagging = tagging if isinstance(tagging, dict) else {}
    value = tagging.get("summary_include_cve_tags", True)
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"false", "no", "off", "0"}:
            return False
        if normalized in {"true", "yes", "on", "1"}:
            return True
        return True
    return bool(value)


def _mask_cve_identifiers(text: str) -> str:
    """Keep CVE context available to the tagger without exposing identifiers."""
    return CVE_PATTERN.sub("CVE-ID", text or "")


def _mentioned_cves(text: str) -> List[str]:
    """Extract every unique CVE from the complete displayed summary text."""
    found: List[str] = []
    seen = set()
    for match in CVE_PATTERN.finditer(text or ""):
        cve = match.group(0).upper()
        if cve not in seen:
            seen.add(cve)
            found.append(cve)
    return found


def _merge_tags(*groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    seen_ids = set()
    seen_names = set()
    for group in groups:
        for tag in group:
            if not isinstance(tag, dict):
                continue
            tag_id = tag.get("id")
            name = str(tag.get("name") or "").strip()
            normalized_name = name.casefold()
            if (tag_id is not None and tag_id in seen_ids) or (
                normalized_name and normalized_name in seen_names
            ):
                continue
            if tag_id is not None:
                seen_ids.add(tag_id)
            if normalized_name:
                seen_names.add(normalized_name)
            merged.append(tag)
    return merged


async def tag_summary_doc(
    *,
    store,
    llm_client,
    config: Dict[str, Any],
    summary_id: str,
    max_tags: int | None = None,
    force: bool = False,
) -> List[Dict[str, Any]]:
    """Classify and persist a summary using the article tagging implementation."""
    sid = str(summary_id or "").strip()
    if not sid:
        return []

    summary_doc = store.get_summary_doc(sid)
    if not isinstance(summary_doc, dict):
        return []

    existing = summary_doc.get("tags")
    if not force and isinstance(existing, list) and existing:
        return [dict(tag) for tag in existing if isinstance(tag, dict)]

    text = _summary_text(summary_doc)
    if not text:
        return []

    tag_manager = TagManager(store, llm_client=llm_client)
    effective_max_tags = _summary_max_tags(config, max_tags)
    include_cve_tags = _summary_include_cve_tags(config)
    title = str(summary_doc.get("title") or "").strip()
    tagger_title = title if include_cve_tags else _mask_cve_identifiers(title)
    tagger_text = text if include_cve_tags else _mask_cve_identifiers(text)
    selected = await tag_manager.generate_tags_for_article(
        llm_client=llm_client,
        article={
            "id": sid,
            "title": tagger_title,
            "content": tagger_text,
        },
        config=config,
        max_tags=effective_max_tags,
    )

    # When CVE tags are enabled, extract them from the complete summary because
    # TagManager's LLM prompt truncates long input and limits its selected result.
    cve_candidates = (
        [
            {
                "name": cve,
                "type": "NAMED_ENTITY",
                "reasoning": "CVE identifier mentioned in the summary.",
            }
            for cve in _mentioned_cves(text)
        ]
        if include_cve_tags
        else []
    )
    cve_tags = (
        await tag_manager.select_tags_for_article_async(
            article_id=sid,
            candidate_tags=cve_candidates,
            allow_new_tags=True,
        )
        if cve_candidates
        else []
    )
    selected = _merge_tags(selected, cve_tags)
    if not include_cve_tags:
        selected = [
            tag
            for tag in selected
            if not isinstance(tag, dict) or not is_cve_tag(tag.get("name"))
        ]
    tags = [
        _stored_tag(tag)
        for tag in selected
        if isinstance(tag, dict) and str(tag.get("name") or "").strip()
    ]

    updated = dict(summary_doc)
    updated["tags"] = tags
    meta = updated.get("meta") if isinstance(updated.get("meta"), dict) else {}
    meta = dict(meta)
    meta["summary_tagged_at"] = int(time.time())
    updated["meta"] = meta
    store.save_summary_doc(updated)
    return tags


async def tag_summary_doc_safe(
    *,
    store,
    llm_client,
    config: Dict[str, Any],
    summary_id: str,
    max_tags: int | None = None,
    logger: logging.Logger | None = None,
) -> List[Dict[str, Any]]:
    """Tag a summary without turning a completed summary job into a failure."""
    try:
        return await tag_summary_doc(
            store=store,
            llm_client=llm_client,
            config=config,
            summary_id=summary_id,
            max_tags=max_tags,
        )
    except Exception as exc:
        (logger or logging.getLogger(__name__)).exception(
            "Could not tag summary_doc %s: %s", summary_id, exc
        )
        return []
