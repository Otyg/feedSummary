from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple


def _extract_section(
    text: str, start: str, end_candidates: List[str]
) -> str:
    if not text:
        return ""
    s = text.find(start)
    if s < 0:
        return ""
    s += len(start)
    end = len(text)
    for cand in end_candidates:
        i = text.find(cand, s)
        if i >= 0:
            end = min(end, i)
    return text[s:end].strip()


def _split_sources_appendix(text: str) -> Tuple[str, str]:
    t = str(text or "").strip()
    if not t:
        return "", ""
    m = re.search(r"(?mi)^\s*##\s*Källor\b", t)
    if not m:
        return t, ""
    i = int(m.start())
    return t[:i].rstrip(), t[i:].strip()


def _split_blocks(text: str) -> List[str]:
    t = str(text or "").strip()
    if not t:
        return []
    parts = re.split(r"\n\s*\n+", t)
    return [p.strip() for p in parts if p and p.strip()]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip().lower())


def _looks_like_feedback_blob(text: str) -> bool:
    t = str(text or "").strip()
    if not t:
        return False
    upper = t.upper()
    if upper.startswith("PASS") or upper.startswith("FAIL"):
        return True
    if re.search(r"(?im)^\s*ISSUES\s*:\s*$", t):
        return True
    if re.search(r"(?im)^\s*-\s*Typ\s*:", t) and re.search(
        r"(?im)^\s*-\s*Åtgärd\s*:", t
    ):
        return True
    return False


def _extract_delete_targets(feedback: str) -> List[str]:
    out: List[str] = []
    for line in str(feedback or "").splitlines():
        m = re.search(r"(?i)\båtgärd\s*:\s*stryk\s*:\s*(.+)$", line.strip())
        if not m:
            continue
        val = str(m.group(1) or "").strip().strip("\"'`")
        if len(val) >= 20:
            out.append(val)
    return out


def _should_drop_block(block: str, delete_targets: List[str]) -> bool:
    if not delete_targets:
        return False
    nb = _norm(block)
    if len(nb) < 20:
        return False
    for target in delete_targets:
        nt = _norm(target)
        if len(nt) < 20:
            continue
        if nt in nb:
            return True
        if len(nt) >= 48 and SequenceMatcher(None, nb, nt).ratio() >= 0.72:
            return True
    return False


def _merge_revised_with_draft(
    *, draft_text: str, revised_text: str, feedback_text: str
) -> str:
    draft_text = str(draft_text or "").strip()
    revised_text = str(revised_text or "").strip()
    if not draft_text:
        return revised_text
    if not revised_text:
        return draft_text

    draft_main, draft_sources = _split_sources_appendix(draft_text)
    revised_main, revised_sources = _split_sources_appendix(revised_text)
    if not revised_main:
        revised_main = revised_text
    if _looks_like_feedback_blob(revised_main):
        return draft_text

    draft_blocks = _split_blocks(draft_main)
    rev_blocks = _split_blocks(revised_main)
    if not draft_blocks or not rev_blocks:
        return revised_text

    used_draft: set[int] = set()
    rev_to_draft: Dict[int, Optional[int]] = {}
    for j, rb in enumerate(rev_blocks):
        best_i: Optional[int] = None
        best_score = 0.0
        nrb = _norm(rb)
        for i, db in enumerate(draft_blocks):
            if i in used_draft:
                continue
            score = SequenceMatcher(None, nrb, _norm(db)).ratio()
            if score > best_score:
                best_score = score
                best_i = i
        if best_i is not None and best_score >= 0.45:
            rev_to_draft[j] = best_i
            used_draft.add(best_i)
        else:
            rev_to_draft[j] = None

    draft_to_rev: Dict[int, int] = {
        int(di): int(rj)
        for rj, di in rev_to_draft.items()
        if isinstance(di, int)
    }

    unmatched_after: Dict[int, List[str]] = {}
    last_anchor = -1
    for j, rb in enumerate(rev_blocks):
        di = rev_to_draft.get(j)
        if isinstance(di, int):
            last_anchor = di
            continue
        unmatched_after.setdefault(last_anchor, []).append(rb)

    delete_targets = _extract_delete_targets(feedback_text)
    merged: List[str] = []
    merged.extend(unmatched_after.get(-1, []))
    for i, db in enumerate(draft_blocks):
        if i in draft_to_rev:
            merged.append(rev_blocks[draft_to_rev[i]])
        else:
            if not _should_drop_block(db, delete_targets):
                merged.append(db)
        merged.extend(unmatched_after.get(i, []))

    merged_main = "\n\n".join([b for b in merged if _norm(b)])
    if not merged_main:
        merged_main = draft_main or revised_main

    final_sources = revised_sources or draft_sources
    if final_sources:
        return f"{merged_main}\n\n{final_sources}".strip()
    return merged_main.strip()


def stabilize_revise_output_from_messages(
    *, messages: List[Dict[str, Any]], raw_reply: str
) -> str:
    reply = str(raw_reply or "").strip()
    if not reply:
        return reply
    if not isinstance(messages, list) or not messages:
        return reply

    user_content = ""
    try:
        for m in reversed(messages):
            if isinstance(m, dict) and str(m.get("role") or "") == "user":
                user_content = str(m.get("content") or "")
                break
    except Exception:
        user_content = ""

    if not user_content:
        return reply

    draft = _extract_section(
        user_content,
        "=== DRAFT SUMMARY ===",
        ["=== DESK-UNDERLAG (källmaterial) ===", "=== DESK-UNDERLAG ==="],
    )
    feedback = _extract_section(
        user_content,
        "=== FEEDBACK (från korrekturläsaren) ===",
        ["=== DRAFT SUMMARY ==="],
    )
    if not draft:
        return reply

    try:
        return _merge_revised_with_draft(
            draft_text=draft, revised_text=reply, feedback_text=feedback
        )
    except Exception:
        return reply
