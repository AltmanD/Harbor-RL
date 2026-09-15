from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from .metadata import stable_hash


@dataclass(frozen=True)
class TerminalTransition:
    """A turn-level terminal transition used by the latent world model."""

    trajectory_id: str
    task_name: str | None
    data_source: str | None
    turn_idx: int
    context_messages: list[dict[str, Any]]
    action_text: str
    feedback_text: str
    next_context_messages: list[dict[str, Any]] | None
    done: bool
    reward: float | None
    status: str | None
    source_path: str
    rollout_id: int | None = None
    train_step: int | None = None
    group_index: int | None = None
    sample_index: int | None = None

    @property
    def transition_id(self) -> str:
        """Stable dedup key hashed from trajectory, turn, action, and feedback."""

        return stable_hash(
            {
                "trajectory_id": self.trajectory_id,
                "turn_idx": self.turn_idx,
                "action_text": self.action_text,
                "feedback_text": self.feedback_text,
            }
        )

    @property
    def has_next(self) -> bool:
        """Whether a successor context exists, so next-state alignment may apply."""

        return bool(self.next_context_messages)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for replay/cache payloads.

        Returns:
            All dataclass fields plus the derived ``transition_id`` and
            ``has_next`` values, so downstream consumers do not recompute them.
        """

        value = asdict(self)
        value["transition_id"] = self.transition_id
        value["has_next"] = self.has_next
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TerminalTransition":
        """Rebuild a transition from a ``to_dict``-style mapping.

        Only declared dataclass fields are read: derived keys such as
        ``transition_id`` are recomputed rather than trusted, and unknown keys
        are ignored so older/newer payloads stay loadable.

        Args:
            value: Mapping of field names, e.g. a ``to_dict`` output.

        Returns:
            The reconstructed transition; fields absent from ``value`` are None.
        """

        fields = cls.__dataclass_fields__
        return cls(**{key: value.get(key) for key in fields})


def _as_messages(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    messages: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            messages.append(dict(item))
        else:
            messages.append({"role": "user", "content": str(item)})
    return messages


def _action_text(turn: dict[str, Any]) -> str:
    parts: list[str] = []
    assistant_output = str(turn.get("assistant_output") or "").strip()
    if assistant_output:
        parts.append(assistant_output)
    for call in turn.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        name = str(call.get("tool_name") or call.get("name") or "tool")
        args = call.get("args", call.get("arguments", {}))
        rendered = json.dumps(args, ensure_ascii=False, sort_keys=True, default=str)
        signature = f"{name}({rendered})"
        # Some trajectory formats keep the parsed tool call outside the raw
        # assistant output.  Avoid duplicating it when it is already present.
        if signature not in assistant_output:
            parts.append(signature)
    return "\n".join(parts)


def _render_tool_result(value: Any) -> str:
    """Render raw stdout/stderr/exit-code fields deterministically."""
    if isinstance(value, dict):
        fields = [
            f"{key}={value[key]}"
            for key in ("stdout", "stderr", "exit_code")
            if value.get(key) is not None
        ]
        if fields:
            return "\n".join(fields)
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return str(value)


def _feedback_text(turn: dict[str, Any], *, status: Any, reward: dict[str, Any]) -> str:
    parts: list[str] = []
    for call in turn.get("tool_calls") or []:
        if not isinstance(call, dict) or call.get("result") is None:
            continue
        name = str(call.get("tool_name") or call.get("name") or "tool")
        parts.append(f"<tool_result name={name}>\n{_render_tool_result(call.get('result'))}\n</tool_result>")
    if parts:
        return "\n\n".join(parts)
    return json.dumps(
        {
            "status": status,
            "score": reward.get("score"),
            "raw_score": reward.get("raw_score"),
        },
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )


def _turn_reward(reward: dict[str, Any], turn_idx: int) -> float | None:
    for row in reward.get("per_turn_scores") or []:
        if isinstance(row, dict) and int(row.get("turn_idx", -1)) == turn_idx and row.get("score") is not None:
            return float(row["score"])
    for key in ("score", "base_score", "raw_score"):
        if reward.get(key) is not None:
            return float(reward[key])
    return None


def transitions_from_seta_trajectory(payload: dict[str, Any], *, source_path: str) -> list[TerminalTransition]:
    """Parse a SETA-native ``traj.json`` payload into turn-level transitions.

    Reward semantics: explicit ``reward.per_turn_scores`` entries win and act
    as dense per-turn rewards.  When they are absent, the episode-level score
    (``score``/``base_score``/``raw_score``) is attached to the final turn
    only, so a single terminal score is treated as sparse return rather than
    being repeated at every turn.

    Args:
        payload: Decoded ``traj.json`` dict with ``info``/``reward``/``turns``.
        source_path: Origin path, stored for provenance and used as fallback
            when the payload carries no trajectory id.

    Returns:
        One transition per turn in turn order; the last turn is marked
        ``done`` and has no ``next_context_messages``.
    """

    info = payload.get("info") if isinstance(payload.get("info"), dict) else {}
    reward = payload.get("reward") if isinstance(payload.get("reward"), dict) else {}
    turns = [turn for turn in (payload.get("turns") or []) if isinstance(turn, dict)]
    trajectory_id = str(info.get("uid") or Path(source_path).parent.name or stable_hash(source_path))
    status = str(info.get("status")) if info.get("status") is not None else None
    transitions: list[TerminalTransition] = []
    for index, turn in enumerate(turns):
        turn_idx = int(turn.get("turn_idx", index))
        turn_reward = _turn_reward(reward, turn_idx)
        # A SETA export commonly stores only the episode score.  Treat that
        # score as a terminal reward rather than accidentally repeating it at
        # every turn; explicit per-turn scores remain dense rewards.
        if not reward.get("per_turn_scores") and index != len(turns) - 1:
            turn_reward = None
        next_messages = None
        if index + 1 < len(turns):
            next_messages = _as_messages(turns[index + 1].get("context_messages"))
        transitions.append(
            TerminalTransition(
                trajectory_id=trajectory_id,
                task_name=str(info.get("task_name")) if info.get("task_name") is not None else None,
                data_source=str(info.get("data_source")) if info.get("data_source") is not None else None,
                turn_idx=turn_idx,
                context_messages=_as_messages(turn.get("context_messages")),
                action_text=_action_text(turn),
                feedback_text=_feedback_text(turn, status=status, reward=reward),
                next_context_messages=next_messages,
                done=index == len(turns) - 1,
                reward=turn_reward,
                status=status,
                source_path=source_path,
                rollout_id=info.get("rollout_id"),
                train_step=info.get("train_step"),
                group_index=info.get("group_index"),
                sample_index=info.get("sample_index"),
            )
        )
    return transitions


def _tb21_observation_text(value: Any) -> str:
    """Render an ATIF observation without depending on Harbor internals.

    ATIF stores tool output in a nested ``observation.results`` list while
    older exports sometimes use a plain list/string.  Keeping this adapter
    dependency-free makes the offline loader usable inside a minimal rjob
    image and gives identical text for repeated runs.
    """

    if isinstance(value, dict):
        results = value.get("results")
        if isinstance(results, list):
            parts: list[str] = []
            for result in results:
                if isinstance(result, dict) and result.get("content") is not None:
                    parts.append(str(result["content"]))
                else:
                    parts.append(json.dumps(result, ensure_ascii=False, sort_keys=True, default=str))
            if parts:
                return "\n\n".join(parts)
        if value.get("content") is not None:
            return str(value["content"])
    if isinstance(value, list):
        return "\n\n".join(_tb21_observation_text(item) for item in value)
    if value is None:
        return ""
    return str(value)


def _tb21_raw_tool_result_text(step: dict[str, Any]) -> str:
    """Render explicit per-call tool results, when an ATIF export preserves them.

    Terminal-Bench ATIF exports commonly contain only ``observation`` (a pane
    capture).  That field is deliberately *not* treated as a tool result.
    Runtime exports that retain ``tool_calls[].result`` are the clean source
    used by ``--require-tool-feedback`` and are wrapped with the same marker as
    SETA trajectories.
    """

    calls: Any = step.get("tool_calls")
    if not isinstance(calls, list) and isinstance(step.get("observation"), dict):
        calls = step["observation"].get("tool_calls")
    if not isinstance(calls, list):
        return ""
    parts: list[str] = []
    for call in calls:
        if not isinstance(call, dict) or call.get("result") is None:
            continue
        name = str(call.get("tool_name") or call.get("name") or "tool")
        rendered = _render_tool_result(call.get("result"))
        parts.append(f"<tool_result name={name}>\n{rendered}\n</tool_result>")
    return "\n\n".join(parts)


def _tb21_feedback_text(step: dict[str, Any]) -> str:
    """Return clean raw tool output when available, otherwise pane text."""

    raw = _tb21_raw_tool_result_text(step)
    if raw:
        return raw
    return _tb21_observation_text(step.get("observation"))


def _tb21_reward_and_status(path: Path) -> tuple[float | None, str | None, str | None]:
    """Read a trial reward next to an ATIF trajectory when available."""

    trial_dir = path.parent.parent if path.parent.name in {"agent", "trajectory"} else path.parent
    reward_path = trial_dir / "verifier" / "reward.txt"
    if reward_path.is_file():
        try:
            value = float(reward_path.read_text(encoding="utf-8").strip().splitlines()[0])
            return value, "completed", str(reward_path)
        except (IndexError, TypeError, ValueError, OSError):
            pass
    result_path = trial_dir / "result.json"
    if result_path.is_file():
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            result = {}
        if isinstance(result, dict):
            for key in ("reward", "score", "verifier_reward", "task_reward"):
                value = result.get(key)
                if isinstance(value, dict):
                    value = value.get("score", value.get("value"))
                try:
                    if value is not None:
                        return float(value), "completed", str(result_path)
                except (TypeError, ValueError):
                    continue
            status = result.get("status") or result.get("state")
            return None, str(status) if status is not None else None, str(result_path)
    return None, None, None


def transitions_from_tb21_trajectory(payload: dict[str, Any], *, source_path: str) -> list[TerminalTransition]:
    """Convert an ATIF-v1 trajectory from Terminal-Bench 2.1.

    A transition is emitted for each ``agent`` step.  The context contains all
    preceding user/assistant/tool messages, while the same step's ATIF
    observation is the feedback target.  This deliberately does not require
    Harbor or the original model service, so complete evaluation artifacts can
    be replayed offline.
    """

    path = Path(source_path)
    steps = [row for row in payload.get("steps", []) if isinstance(row, dict)]
    if not steps:
        return []
    reward, status, reward_source = _tb21_reward_and_status(path)
    agent_meta = payload.get("agent") if isinstance(payload.get("agent"), dict) else {}
    trial_dir = path.parent.parent if path.parent.name in {"agent", "trajectory"} else path.parent
    trajectory_id = str(payload.get("session_id") or trial_dir.name or stable_hash(source_path))
    task_name: str | None = None
    result_path = trial_dir / "result.json"
    if result_path.is_file():
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            result = {}
        if isinstance(result, dict) and result.get("task_name") is not None:
            task_name = str(result["task_name"])
    if task_name is None:
        task_name = str(trial_dir.name)
    messages: list[dict[str, Any]] = []
    agent_positions = [
        i
        for i, row in enumerate(steps)
        if str(row.get("source", "")).lower() in {"agent", "assistant"}
        and str(row.get("message") or row.get("content") or "").strip()
        and _tb21_feedback_text(row)
    ]
    transitions: list[TerminalTransition] = []
    cursor = 0
    for turn_idx, step_index in enumerate(agent_positions):
        # ATIF records the user prompt as a separate step before the first
        # agent action.  Fold those preceding non-agent steps into h_t so the
        # encoder sees the same conversation that produced the action.  The
        # previous agent's observation is already appended below, so cursor
        # advances past each emitted action and avoids duplicate messages.
        while cursor < step_index:
            prior = steps[cursor]
            source = str(prior.get("source", "")).lower()
            content = prior.get("message") or prior.get("content")
            if source in {"user", "system", "developer"} and content:
                messages.append({"role": source, "content": str(content)})
            elif source in {"tool", "environment", "observation"}:
                observation = _tb21_observation_text(prior.get("observation", content))
                if observation:
                    messages.append({"role": "tool", "content": observation})
            cursor += 1
        step = steps[step_index]
        action_text = str(step.get("message") or step.get("content") or "").strip()
        feedback_text = _tb21_feedback_text(step)
        next_messages = list(messages)
        next_messages.extend(
            [
                {"role": "assistant", "content": action_text},
                {"role": "tool", "content": feedback_text},
            ]
        )
        is_last_action = turn_idx == len(agent_positions) - 1
        transitions.append(
            TerminalTransition(
                trajectory_id=trajectory_id,
                task_name=task_name,
                data_source="tb21",
                turn_idx=turn_idx,
                context_messages=list(messages),
                action_text=action_text,
                feedback_text=feedback_text,
                next_context_messages=None if is_last_action else next_messages,
                done=is_last_action,
                # ATIF's verifier score is an episode-level terminal reward.
                reward=reward if is_last_action else None,
                status=status or "unknown",
                source_path=source_path,
                rollout_id=None,
                train_step=None,
                group_index=None,
                sample_index=None,
            )
        )
        messages = next_messages
        cursor = step_index + 1
    # ``reward_source`` is read for provenance validation; the transition
    # schema intentionally stays independent from neighboring result files.
    del agent_meta, reward_source
    return transitions


def _messages_from_record(record: dict[str, Any]) -> list[dict[str, Any]]:
    if record.get("context_messages"):
        return _as_messages(record["context_messages"])
    text = record.get("context_text")
    if not text:
        return []
    try:
        parsed = json.loads(str(text))
    except (TypeError, json.JSONDecodeError):
        return [{"role": "user", "content": str(text)}]
    if isinstance(parsed, dict) and "context_messages" in parsed:
        return _as_messages(parsed["context_messages"])
    return _as_messages(parsed)


def transition_from_world_model_record(record: dict[str, Any], *, source_path: str) -> TerminalTransition:
    """Build one transition from a world-model record (JSONL row or ``.pt`` item).

    Alternative key spellings from different exporters are accepted
    (``uid``/``trajectory_id``, ``next_observation_text``/``feedback_text``,
    structured ``context_messages`` or serialized ``context_text``) so replay
    checkpoints and offline record files share a single ingestion path.

    Args:
        record: Record dict; unrecognized keys are ignored.
        source_path: Origin path, stored for provenance.

    Returns:
        The populated transition; ``reward`` stays None when the record has
        no ``reward_score``.
    """

    status = str(record.get("status")) if record.get("status") is not None else None
    reward = record.get("reward_score")
    next_context_messages = None
    if record.get("next_context_messages"):
        next_context_messages = _as_messages(record.get("next_context_messages"))
    elif record.get("next_context_text"):
        next_context_messages = _messages_from_record({"context_text": record.get("next_context_text")})
    return TerminalTransition(
        trajectory_id=str(record.get("uid") or record.get("trajectory_id") or stable_hash(record)),
        task_name=str(record.get("task_name")) if record.get("task_name") is not None else None,
        data_source=str(record.get("data_source")) if record.get("data_source") is not None else None,
        turn_idx=int(record.get("turn_idx", 0) or 0),
        context_messages=_messages_from_record(record),
        action_text=str(record.get("action_text") or ""),
        feedback_text=str(record.get("next_observation_text") or record.get("feedback_text") or ""),
        next_context_messages=next_context_messages,
        done=bool(record.get("done", False)),
        reward=float(reward) if reward is not None else None,
        status=status,
        source_path=source_path,
        rollout_id=record.get("rollout_id"),
        train_step=record.get("train_step"),
        group_index=record.get("group_index"),
        sample_index=record.get("sample_index"),
    )


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                value = json.loads(line)
                if isinstance(value, dict):
                    yield value


def _load_pt_records(path: Path) -> list[dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "world_model_replay" in payload:
        payload = payload["world_model_replay"]
    if isinstance(payload, dict):
        records = payload.get("records") or payload.get("items")
        if records is not None:
            return [dict(row) for row in records if isinstance(row, dict)]
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, dict)]
    raise ValueError(f"No world-model replay records found in {path}")


def _discover_paths(root: Path, source: str) -> list[Path]:
    if root.is_file():
        return [root]
    if not root.is_dir():
        raise FileNotFoundError(f"World-model input does not exist: {root}")
    source = source.lower()
    if source == "tb21":
        return sorted(root.rglob("trajectory.json"))
    if source == "seta":
        return sorted(root.rglob("traj.json"))
    if source == "records":
        return sorted(root.rglob("*.jsonl"))
    if source == "replay":
        return sorted(list(root.rglob("*.pt")) + list(root.rglob("*.pth")))
    if source not in {"auto", "mixed"}:
        raise ValueError(f"Unknown data source {source!r}; expected auto, tb21, seta, records, replay, or mixed")
    # Explicit order is important: tb2.1 is the primary offline source and
    # records/replay are only considered after native trajectory formats.  Do
    # not eagerly recurse through large checkpoint/replay trees when native
    # trajectories already exist; callers can pass a separate supplement root
    # when they intentionally want those lower-priority formats mixed in.
    native_paths = sorted(root.rglob("trajectory.json")) + sorted(root.rglob("traj.json"))
    if native_paths:
        return native_paths
    paths: list[Path] = []
    for pattern in ("*.jsonl", "*.pt", "*.pth"):
        paths.extend(sorted(root.rglob(pattern)))
    return paths


def _iter_discovered_paths(root: Path, source: str) -> Iterable[Path]:
    """Yield paths lazily so lower-priority formats are scanned only if needed."""

    if root.is_file():
        yield root
        return
    if source.lower() in {"auto", "mixed"}:
        for name in ("tb21", "seta", "records", "replay"):
            yield from _discover_paths(root, name)
        return
    yield from _discover_paths(root, source)


def _load_path(path: Path) -> list[TerminalTransition]:
    if path.name == "trajectory.json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "steps" in payload:
            return transitions_from_tb21_trajectory(payload, source_path=str(path))
    if path.suffix in {".pt", ".pth"}:
        rows = _load_pt_records(path)
        return [transition_from_world_model_record(row, source_path=str(path)) for row in rows]
    if path.suffix == ".jsonl":
        return [transition_from_world_model_record(row, source_path=str(path)) for row in _iter_jsonl(path)]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return []
    return transitions_from_seta_trajectory(payload, source_path=str(path))


def build_data_manifest(transitions: Sequence[TerminalTransition], *, requested_source: str = "auto") -> dict[str, Any]:
    """Return stable, JSON-serialisable dataset provenance and coverage stats."""

    trajectory_ids = {row.trajectory_id for row in transitions}
    task_names = {row.task_name for row in transitions if row.task_name}
    sources = Counter(row.data_source or "unknown" for row in transitions)
    return {
        "schema_version": "lwm_offline_manifest_v1",
        "requested_source": requested_source,
        "transition_count": len(transitions),
        "trajectory_count": len(trajectory_ids),
        "task_count": len(task_names),
        "terminal_transition_count": sum(row.done for row in transitions),
        "nonterminal_transition_count": sum(not row.done for row in transitions),
        "reward_count": sum(row.reward is not None for row in transitions),
        "tool_feedback_count": sum("<tool_result" in row.feedback_text for row in transitions),
        "source_histogram": dict(sorted(sources.items())),
        "trajectory_ids": sorted(trajectory_ids),
        "task_names": sorted(task_names),
        "source_paths": sorted({row.source_path for row in transitions}),
    }


def load_terminal_transitions(
    input_path: str | Path,
    *,
    max_trajectories: int | None = None,
    max_transitions: int | None = None,
    require_tool_feedback: bool = False,
    data_source: str = "auto",
    supplement_inputs: Sequence[str | Path] | None = None,
    min_turns: int = 0,
    include_terminal: bool = True,
) -> list[TerminalTransition]:
    """Load tb2.1/SETA/replay transitions through one configurable interface.

    ``input_path`` is read first.  Supplement paths are then considered in the
    supplied order and only fill still-unseen trajectories/transitions.  This
    makes ``data_source=auto`` deterministic while preserving tb2.1 priority.
    """

    root = Path(input_path).expanduser()
    requested = [root, *[Path(item).expanduser() for item in (supplement_inputs or [])]]
    transitions: list[TerminalTransition] = []
    seen_ids: set[str] = set()
    seen_trajectories: set[str] = set()
    for index, candidate in enumerate(requested):
        if max_trajectories is not None and len(seen_trajectories) >= max(0, int(max_trajectories)):
            break
        if not candidate.exists():
            if index == 0:
                raise FileNotFoundError(f"World-model input does not exist: {candidate}")
            # Optional supplement roots are allowed to be absent on a worker
            # that only mounted the primary tb2.1 evaluation tree.
            continue
        source = data_source if index == 0 else "auto"
        # Keep discovery lazy: once max_trajectories/max_transitions is met,
        # no lower-priority checkpoint tree is recursively scanned.
        for path in _iter_discovered_paths(candidate, source):
            if max_trajectories is not None and len(seen_trajectories) >= max(0, int(max_trajectories)):
                break
            try:
                batch = _load_path(path)
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                # A run directory may contain partial/corrupt artifacts.  Skip
                # only that file; explicit single-file inputs still fail below if
                # they produce no valid transition.
                batch = []
            for transition in batch:
                if transition.trajectory_id not in seen_trajectories:
                    trajectory_size = sum(row.trajectory_id == transition.trajectory_id for row in batch)
                    if min_turns > 0 and trajectory_size < int(min_turns):
                        continue
                    if max_trajectories is not None and len(seen_trajectories) >= max(0, int(max_trajectories)):
                        break
                    seen_trajectories.add(transition.trajectory_id)
                if require_tool_feedback and "<tool_result" not in transition.feedback_text:
                    continue
                if not include_terminal and transition.done:
                    continue
                if not transition.action_text or not transition.feedback_text:
                    continue
                if transition.transition_id in seen_ids:
                    continue
                seen_ids.add(transition.transition_id)
                transitions.append(transition)
                if max_transitions is not None and len(transitions) >= int(max_transitions):
                    return transitions
    return transitions
