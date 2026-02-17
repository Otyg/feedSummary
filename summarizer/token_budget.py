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

from __future__ import annotations
from typing import Dict, List, Tuple


def estimate_tokens(text: str) -> int:
    # konservativ approximation (svenska/URL/markup tenderar att bli fler tokens)
    return max(1, int(len(text) / 3.6))


def messages_to_text(messages: List[Dict[str, str]]) -> str:
    # grov approximation av chat-format overhead
    out = []
    for m in messages:
        out.append(m.get("role", "user") + ":\n" + (m.get("content") or ""))
    return "\n\n".join(out)


def enforce_budget(
    messages: List[Dict[str, str]],
    *,
    max_context_tokens: int,
    max_output_tokens: int,
    safety_margin_tokens: int,
) -> Tuple[List[Dict[str, str]], int, int]:
    """
    Returnerar (ev. reducerade messages, est_prompt_tokens, budget_tokens).

    Strategi:
      - räkna estimerade prompt tokens
      - om över budget: kapa content i sista user-meddelandet (där din corpus ligger)
    """
    budget = max_context_tokens - max_output_tokens - safety_margin_tokens
    if budget < 256:
        budget = 256  # fail-safe

    text = messages_to_text(messages)
    est = estimate_tokens(text)

    if est <= budget:
        return messages, est, budget

    # kapa bara sista user content (där bulk-data finns)
    # om du har flera user messages kan du göra mer avancerat.
    reduced = [dict(m) for m in messages]
    # hitta sista user
    idx = None
    for i in range(len(reduced) - 1, -1, -1):
        if reduced[i].get("role") == "user":
            idx = i
            break
    if idx is None:
        return messages, est, budget

    content = reduced[idx].get("content") or ""
    # proportionell kapning baserat på tokens
    # tokens ~ chars/3.6 => chars ~ tokens*3.6
    target_chars = int(budget * 3.6)

    # lämna gärna början + slut med käll-länkar; men enklast: klipp rakt av
    if len(content) > target_chars:
        reduced[idx]["content"] = (
            content[:target_chars] + "\n\n[TRUNCATED FOR CONTEXT WINDOW]\n"
        )

    new_est = estimate_tokens(messages_to_text(reduced))
    return reduced, new_est, budget
