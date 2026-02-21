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
#

from summarizer.helpers import text_clip


from typing import Any, Dict, List, Optional, Tuple

import logging

logger = logging.getLogger(__name__)


def batch_articles(
    articles: List[dict],
    max_chars_per_batch: int,
    max_articles_per_batch: int,
    article_clip_chars: int = 2500,
) -> List[List[dict]]:
    batches: List[List[dict]] = []
    current: List[dict] = []
    current_chars = 0

    for a in articles:
        per_article_text = text_clip(a.get("text", ""), article_clip_chars)
        estimated = (
            len(per_article_text)
            + len(a.get("title", ""))
            + len(a.get("url", ""))
            + 200
        )

        if current and (
            current_chars + estimated > max_chars_per_batch
            or len(current) >= max_articles_per_batch
        ):
            batches.append(current)
            current = []
            current_chars = 0

        a2 = dict(a)
        a2["text"] = per_article_text
        current.append(a2)
        current_chars += estimated

    if current:
        batches.append(current)

    return batches


# ----------------------------
# Prompt-too-long helpers + stable resume helpers
# ----------------------------
class PromptTooLongStructural(Exception):
    def __init__(self, overflow_tokens: int):
        super().__init__(f"prompt too long (structural), overflow={overflow_tokens}")
        self.overflow_tokens = overflow_tokens


def _choose_trim_action(overflow_tokens: int, structural_threshold: int) -> str:
    if overflow_tokens <= 200:
        return "word_trim"
    if overflow_tokens <= structural_threshold:
        return "drop_one_article"
    return "drop_multiple_articles"


def trim_text_tail_by_words(
    text: str, remove_tokens: int, *, chars_per_token: float
) -> str:
    """
    Tar bort från slutet men alltid på whitespace (ordgräns).
    """
    s = text or ""
    if not s:
        return s

    remove_chars = int(max(1, remove_tokens) * chars_per_token)
    if remove_chars >= len(s):
        return ""

    target = max(0, len(s) - remove_chars)
    cut = max(
        s.rfind(" ", 0, target), s.rfind("\n", 0, target), s.rfind("\t", 0, target)
    )
    if cut <= 0:
        cut = target

    return s[:cut].rstrip() + "\n\n[TRUNCATED FOR CONTEXT WINDOW]\n"


def _trim_last_user_word_boundary(
    messages: List[Dict[str, str]], remove_tokens: int, *, chars_per_token: float
) -> List[Dict[str, str]]:
    out = [dict(m) for m in messages]
    idx = None
    for i in range(len(out) - 1, -1, -1):
        if out[i].get("role") == "user":
            idx = i
            break
    if idx is None:
        return out
    content = out[idx].get("content") or ""
    out[idx]["content"] = trim_text_tail_by_words(
        content, remove_tokens, chars_per_token=chars_per_token
    )
    return out


def _estimate_article_chars(a: dict) -> int:
    return (
        len(a.get("text", "") or "")
        + len(a.get("title", "") or "")
        + len(a.get("url", "") or "")
        + 200
    )


def _batch_chars(batch: List[dict]) -> int:
    return sum(_estimate_article_chars(x) for x in batch)


def _can_fit_in_batch(
    batch: List[dict], a: dict, *, max_chars_per_batch: int, max_articles_per_batch: int
) -> bool:
    if max_articles_per_batch and len(batch) >= max_articles_per_batch:
        return False
    return (_batch_chars(batch) + _estimate_article_chars(a)) <= max_chars_per_batch


def _move_article_to_tail_batch(
    batches: List[List[dict]],
    a: dict,
    *,
    max_chars_per_batch: int,
    max_articles_per_batch: int,
    avoid_batch: Optional[List[dict]] = None,
) -> None:
    """
    Flytta till sista batch om plats, annars ny batch.
    Viktigt: undvik att lägga tillbaka i samma batch (tail-loop).
    """
    if not batches:
        batches.append([a])
        return

    last = batches[-1]
    if avoid_batch is not None and last is avoid_batch:
        batches.append([a])
        return

    if _can_fit_in_batch(
        last,
        a,
        max_chars_per_batch=max_chars_per_batch,
        max_articles_per_batch=max_articles_per_batch,
    ):
        last.append(a)
    else:
        batches.append([a])


def _batch_article_ids_map(batches_local: List[List[dict]]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for i, b in enumerate(batches_local, start=1):
        out[str(i)] = [str(a.get("id", "")) for a in b if a.get("id")]
    return out


def _done_batches_payload(
    done_map_local: Dict[int, str], batches_local: List[List[dict]]
) -> Dict[str, Dict[str, Any]]:
    ids_map = _batch_article_ids_map(batches_local)
    payload: Dict[str, Dict[str, Any]] = {}
    for k, v in sorted(done_map_local.items()):
        sk = str(k)
        payload[sk] = {"article_ids": ids_map.get(sk, []), "summary": v}
    return payload


def _done_map_from_done_batches(cp_done_batches: Dict[str, Any]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    if not isinstance(cp_done_batches, dict):
        return out
    for k, entry in cp_done_batches.items():
        try:
            idx = int(k)
        except Exception:
            continue
        if isinstance(entry, dict) and isinstance(entry.get("summary"), str):
            out[idx] = entry["summary"]
    return out


def _build_batches_from_checkpoint(
    batch_article_ids: Dict[str, Any],
    all_articles: List[dict],
    *,
    clip_chars: int,
) -> List[List[dict]]:
    """
    Återskapa batch-indelning EXAKT från checkpointens batch_article_ids.
    """
    by_id: Dict[str, dict] = {}
    for a in all_articles:
        aid = a.get("id")
        if aid:
            by_id[str(aid)] = a

    def key_int(k: str) -> int:
        try:
            return int(k)
        except Exception:
            return 10**9

    rebuilt: List[List[dict]] = []
    missing: List[str] = []

    for k in sorted(batch_article_ids.keys(), key=key_int):
        ids = batch_article_ids.get(k)
        if not isinstance(ids, list):
            continue

        batch: List[dict] = []
        for aid in ids:
            aid_s = str(aid)
            a = by_id.get(aid_s)
            if not a:
                missing.append(aid_s)
                continue
            a2 = dict(a)
            a2["text"] = text_clip(a2.get("text", ""), clip_chars)
            batch.append(a2)

        if batch:
            rebuilt.append(batch)

    if missing:
        logger.warning(
            "Resume: %d article_ids saknas i store (hoppas över). Ex: %s",
            len(missing),
            ", ".join(missing[:5]),
        )

    if not rebuilt:
        raise RuntimeError(
            "Resume: batch_article_ids fanns men inga batcher kunde återskapas."
        )

    return rebuilt


def _budgeted_meta_user(
    *,
    prompts: Dict[str, str],
    batch_summaries: List[Tuple[int, str]],
    sources_text: str,
    budget_tokens: int,
    chars_per_token: float,
    lookback:str,
) -> str:
    """
    Bygg meta-user inom en *explicit* tokenbudget.
    Skalar ner via clip-levels, käll-clip och decimering.
    """

    def est_tokens(s: str) -> int:
        return max(1, int(len(s) / chars_per_token))

    def render(batch_block: str, src: str) -> str:
        return prompts["meta_user_template"].format(
            batch_summaries=batch_block,
            sources_list=src,
            lookback=lookback or "okänt tidsfönster",
        )

    # Aggressivare stegar än innan (särskilt decimations)
    sources_levels = [len(sources_text), 6000, 3500, 2000, 1200, 700]
    clip_levels = [4200, 3200, 2400, 1800, 1200, 900, 700, 500, 350, 250]
    decimations = [1, 2, 3, 4, 6, 8, 12]  # 1=alla batcher, 2=varannan, ...

    summaries_desc = sorted(batch_summaries, key=lambda x: x[0], reverse=True)

    for src_lim in sources_levels:
        src2 = (
            sources_text
            if len(sources_text) <= src_lim
            else (sources_text[:src_lim].rstrip() + "…")
        )

        for dec in decimations:
            subset = summaries_desc[::dec]

            for clip_n in clip_levels:
                parts: List[str] = []

                for i, s in subset:
                    s2 = (s or "").strip()
                    if len(s2) > clip_n:
                        s2 = s2[:clip_n].rstrip() + "…"

                    candidate_parts = parts + [f"Batch {i}:\n{s2}"]
                    candidate_block = "\n\n====================\n\n".join(
                        candidate_parts
                    )
                    candidate_user = render(candidate_block, src2)

                    if est_tokens(candidate_user) <= budget_tokens:
                        parts = candidate_parts
                    else:
                        if parts:
                            break
                        continue

                if parts:
                    return render("\n\n====================\n\n".join(parts), src2)

    # Sista utväg: utan batch-summaries
    return render(
        "[Inga batch-summaries kunde inkluderas inom context-budget.]",
        (sources_text[:700].rstrip() + "…")
        if len(sources_text) > 700
        else sources_text,
    )


def _est_user_tokens(s: str, chars_per_token: float) -> int:
    return max(1, int(len(s) / chars_per_token))


def build_messages_for_batch(
    *,
    prompts: Dict[str, str],
    batch_index: int,
    batch_total: int,
    batch_items: List[dict],
) -> List[Dict[str, str]]:
    """
    Shared batch prompt builder (used by summarizer + prompt_lab).

    Returns a chat message list compatible with LLMClient.chat.
    """
    parts: List[str] = []
    for i, a in enumerate(batch_items, start=1):
        parts.append(
            f"[{i}] {a.get('title', '')}\n"
            f"Källa: {a.get('source', '')}\n"
            f"Publicerad: {a.get('published', '')}\n"
            f"URL: {a.get('url', '')}\n\n"
            f"{a.get('text', '')}"
        )

    corpus = "\n\n---\n\n".join(parts)
    user_content = prompts["batch_user_template"].format(
        batch_index=batch_index,
        batch_total=batch_total,
        articles_corpus=corpus,
    )

    return [
        {"role": "system", "content": prompts["batch_system"]},
        {"role": "user", "content": user_content},
    ]
