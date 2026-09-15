import json
import random
from typing import Any, Dict, List, Optional


PRM_SYSTEM = """You are an evaluator for a terminal agent.
You are provided with:
1) the agent's task instruction,
2) the interaction history, and
3) the agent's most recent step to evaluate.
"""

USER_INSTRUCTION = """
Evaluate ONLY the single most recent step using the information above.


Assign a score of +1 if ALL of the following are true:
- The current assistant message is a correct/helpful step that advances the task;
- The tool-call format is valid;
- Tool usage is appropriate for the step;
- Tool results (if any) are consistent with making progress.

Otherwise assign a score of -1, for example if:
- The step is incorrect, misleading, or does not advance the task;
- Tool-call format is broken (invalid JSON / parse error);
- Tool usage is clearly wrong or irrelevant;
- Tool results show failure or clearly no progress.

Think carefully, then provide your reasoning and put the final score in \\boxed{}.
"""


import re

_PRM_BOXED_PATTERN = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", re.DOTALL)
_PRM_STRICT_NUMBER_PATTERN = re.compile(r"^\s*([-+]?\d+(?:\.\d+)?)\s*$")


def _extract_prm_sign_from_text(text: str) -> int:
    if not text:
        return 0
    match = _PRM_BOXED_PATTERN.search(text)
    if not match:
        return 0
    boxed_content = match.group(1).strip()
    strict_number_match = _PRM_STRICT_NUMBER_PATTERN.fullmatch(boxed_content)
    if not strict_number_match:
        return 0
    try:
        value = float(strict_number_match.group(1))
    except ValueError:
        return 0
    if abs(value - 1.0) < 1e-9:
        return 1
    if abs(value + 1.0) < 1e-9:
        return -1
    return 0


def _truncate(text: str, limit: int = 2000) -> str:
    if not text:
        return ""
    return text if len(text) <= limit else (text[:limit] + "...<truncated>")


class TerminalPRMAgent:
    """
    PRM agent.

    history_mode:
      - "last": last k turns
      - "random": random sample k turns
      - "head_tail": first k turns + last k turns
    """

    def __init__(
        self,
        *,
        sglang_client,
        task_instruction: str,
        history_k: int = 3,
        history_mode: str = "head_tail",
        head_k: int = 2,
        tail_k: int = 2,
        history_include_assistant: bool = False,
        current_truncate: int = 2000,
        history_truncate: int = 5000,  # characters not tokens
    ):
        self._sglang_client = sglang_client
        self.task_instruction = task_instruction

        self.history_k = history_k
        self.history_mode = history_mode
        self.head_k = head_k
        self.tail_k = tail_k
        self.history_include_assistant = history_include_assistant

        self.current_truncate = current_truncate
        self.history_truncate = history_truncate

        self._history: Dict[int, Dict[str, Any]] = {}

    def record_model_turn(
        self,
        turn_idx: int,
        *,
        assistant_text: str,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        parse_error_recorded: bool = False,
        finish_reason: Optional[str] = None,
    ) -> None:
        rec = self._history.setdefault(turn_idx, {})
        rec["assistant_text"] = assistant_text
        rec["tool_calls"] = tool_calls
        rec["parse_error_recorded"] = bool(parse_error_recorded)
        rec["finish_reason"] = finish_reason
        rec.setdefault("tool_results", [])

    def record_tool_result(
        self, turn_idx: int, tool_call_request, raw_result: Any
    ) -> None:
        rec = self._history.setdefault(turn_idx, {})
        lst = rec.setdefault("tool_results", [])
        lst.append(
            {
                "name": tool_call_request.tool_name,
                "args": tool_call_request.args,
                "result": raw_result,
            }
        )

    def get_history(self, current_turn_idx: int) -> List[Dict[str, Any]]:
        prev = sorted(t for t in self._history.keys() if t < current_turn_idx)
        if not prev:
            return []

        if self.history_mode == "last":
            if self.history_k <= 0:
                return []
            chosen = prev[-self.history_k :]

        elif self.history_mode == "random":
            if self.history_k <= 0:
                return []
            k = min(self.history_k, len(prev))
            chosen = random.sample(prev, k=k)
            chosen = sorted(chosen)

        elif self.history_mode == "head_tail":
            head = prev[: max(0, self.head_k)]
            tail = prev[-max(0, self.tail_k) :] if self.tail_k > 0 else []
            # de-dup while preserving order
            seen = set()
            chosen = []
            for t in head + tail:
                if t not in seen:
                    seen.add(t)
                    chosen.append(t)

        else:
            raise ValueError(f"Invalid history mode: {self.history_mode}")

        hist: List[Dict[str, Any]] = []
        for t in chosen:
            r = self._history.get(t, {})
            item = {
                "turn_idx": t,
                "assistant_text": (
                    _truncate(r["assistant_text"], self.history_truncate)
                    if self.history_include_assistant
                    else "[OMITTED]"
                ),
                "tool_calls": r["tool_calls"],
                "tool_results": r["tool_results"],
                "parse_error_recorded": r["parse_error_recorded"],
            }
            hist.append(item)
        return hist

    def _build_messages(self, turn_idx: int) -> List[Dict[str, str]]:
        cur = self._history.get(turn_idx, {})
        history = self.get_history(turn_idx)

        payload = {
            "task_instruction": self.task_instruction,
            "history": history,
            "current": {
                "turn_idx": turn_idx,
                "assistant_text": _truncate(
                    cur["assistant_text"], self.current_truncate
                ),
                "tool_calls": cur["tool_calls"],
                "tool_results": cur["tool_results"],
                "parse_error_recorded": cur["parse_error_recorded"],
            },
        }

        return [
            {"role": "system", "content": PRM_SYSTEM},
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False)
                + "\n\n"
                + USER_INSTRUCTION,
            },
        ]

    async def judge_turn(self, turn_idx: int) -> int:
        messages = self._build_messages(turn_idx)
        _cc, interaction = await self._sglang_client.generate_turn(
            messages=messages,
            tools=None,
            turn_idx=turn_idx,
        )
        text = interaction.output_text
        score = _extract_prm_sign_from_text(text[-50:])
        return text, score
