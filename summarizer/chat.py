from llmClient import LLMClient
from summarizer.batching import (
    PromptTooLongStructural,
    _choose_trim_action,
    _trim_last_user_word_boundary,
)
from summarizer.helpers import _extract_overflow_tokens
from summarizer.token_budget import enforce_budget
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


async def chat_guarded(
    *,
    llm: LLMClient,
    messages: List[Dict[str, str]],
    temperature: float,
    max_ctx: int,
    max_out: int,
    margin: int,
    chars_per_token: float,
    max_attempts: int,
    structural_threshold: int,
) -> str:
    """
    Shared guarded chat helper.

    - enforce_budget (best effort)
    - om overflow <= 200 => trim sista user (word boundary) och retry
    - annars => raise PromptTooLongStructural för batch/meta-logik
    """
    attempt = 1
    current, est, budget = enforce_budget(
        messages,
        max_context_tokens=max_ctx,
        max_output_tokens=max_out,
        safety_margin_tokens=margin,
    )
    logger.info(f"LLM budget: est_prompt_tokens={est} budget_tokens={budget}")

    while True:
        try:
            return await llm.chat(current, temperature=temperature)
        except Exception as e:
            msg = str(e).lower()
            overflow = _extract_overflow_tokens(e)

            if (
                "prompt too long" in msg
                or "max context" in msg
                or "context length" in msg
            ):
                if attempt >= max_attempts:
                    raise

                if overflow is None:
                    # okänt overflow: trim schablon
                    current = _trim_last_user_word_boundary(
                        current, 2048, chars_per_token=chars_per_token
                    )
                    attempt += 1
                    continue

                overflow_i = int(overflow)
                action = _choose_trim_action(overflow_i, structural_threshold)

                if action == "word_trim":
                    remove_tokens = overflow_i + 1024
                    logger.warning(
                        "LLM prompt too long: overflow=%s action=word_trim attempt=%s/%s",
                        overflow_i,
                        attempt,
                        max_attempts,
                    )
                    current = _trim_last_user_word_boundary(
                        current, remove_tokens, chars_per_token=chars_per_token
                    )
                    attempt += 1
                    continue

                raise PromptTooLongStructural(overflow_i)

            raise
