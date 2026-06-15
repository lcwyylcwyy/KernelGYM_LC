from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import pandas as pd

from ncu_tiers import FULL_METRIC_SPACE, PACKS, resolve_request

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = (
    ROOT
    / "data/drkernel-validation-data/validation_data_thinking_matmul_precision_mini10.parquet"
)
DEFAULT_SERVER_URL = "http://172.17.2.155:8002"
DEFAULT_EVAL_ROOT = ROOT / "drkernel/kernel/scripts/eval"


CODE_FENCE_RE = re.compile(r"```(?:\w+)?\s*\n?(?P<code>.*?)```", re.S)
ROLE_PREFIX_RE = re.compile(r"^(?:user|assistant|system)\n", re.I)
VISIBLE_NOTES_INSTRUCTION = """

Before the Python code block, include a concise visible section exactly titled `Optimization notes:` with 3-6 bullets covering the optimization strategy, expected bottleneck, and any tradeoffs. Do not include hidden chain-of-thought; keep this as a short user-facing engineering summary.
"""


def strip_legacy_role_prefix(text: str) -> str:
    return ROLE_PREFIX_RE.sub("", text, count=1)


def prompt_text_from_cell(prompt_cell: Any) -> str:
    if hasattr(prompt_cell, "tolist"):
        prompt_cell = prompt_cell.tolist()
    if isinstance(prompt_cell, dict):
        return str(prompt_cell.get("content", ""))
    if isinstance(prompt_cell, list):
        parts: list[str] = []
        for item in prompt_cell:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "\n\n".join(parts)
    return str(prompt_cell)


def extract_python_code(text: str) -> str:
    for match in reversed(list(CODE_FENCE_RE.finditer(text))):
        code = match.group("code").strip()
        if "class ModelNew" in code:
            return code

    if "class ModelNew" in text:
        return text.strip()

    raise ValueError(
        "Claude response did not contain a Python block with class ModelNew"
    )


def extract_optimization_notes(text: str) -> str:
    before_code = text.split("```", 1)[0].strip()
    if not before_code:
        return (
            "Optimization notes:\n"
            "- No visible optimization notes were found before the code block."
        )

    marker = "optimization notes:"
    marker_idx = before_code.lower().find(marker)
    if marker_idx >= 0:
        return before_code[marker_idx:].strip()
    return "Optimization notes:\n" + before_code


def prompt_with_visible_notes_instruction(prompt: str) -> str:
    return prompt.rstrip() + VISIBLE_NOTES_INSTRUCTION


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return json_safe(value.tolist())
        except Exception:
            pass
    return str(value)


@dataclass(frozen=True)
class WorkItem:
    row_index: int
    problem_id: int
    name: str
    prompt_text: str
    reference_code: str
    sample_id: int

    @property
    def key(self) -> str:
        return f"p{self.problem_id}_s{self.sample_id}"


def _metric_str(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "None"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def score_from_eval(eval_result: dict[str, Any]) -> float:
    if eval_result.get("correctness") is not True:
        return 0.0
    speedup = _as_float(eval_result.get("speedup"))
    if speedup is None:
        return 1.0 if eval_result.get("compiled") is True else 0.0
    return round(speedup, 6)


# drkernel reward (calculate_reward_speedup): selection metric for STTS history,
# matching drkernel-14b's best-K selection. reward = init_correct_weight*correct
# + init_performance_weight*min(speedup, upper_bound). speedup is capped, so any
# correct turn >= upper_bound speedup ties — selection then favors correctness +
# the cap, NOT raw speedup (35x and 3x are equal-reward). Failed/decoy => penalty.
STTS_REWARD_CORRECT_WEIGHT = float(os.getenv("KG_REWARD_CORRECT_WEIGHT", "0.5"))
STTS_REWARD_PERF_WEIGHT = float(os.getenv("KG_REWARD_PERF_WEIGHT", "0.5"))
STTS_REWARD_SPEEDUP_UPPER = float(os.getenv("KG_REWARD_SPEEDUP_UPPER", "3.0"))
STTS_REWARD_SPEEDUP_LOWER = float(os.getenv("KG_REWARD_SPEEDUP_LOWER", "0.0"))


def reward_from_eval(eval_result: dict[str, Any]) -> float:
    """drkernel-aligned reward used as the STTS best-K selection metric."""
    if eval_result.get("decoy_kernel") is True:
        return 0.0
    if eval_result.get("correctness") is not True:
        return 0.0
    speedup = _as_float(eval_result.get("speedup")) or 0.0
    reward_speedup = min(speedup, STTS_REWARD_SPEEDUP_UPPER)
    if reward_speedup < STTS_REWARD_SPEEDUP_LOWER:
        reward_speedup = 0.0
    correctness = 1.0  # already gated to correct above
    return round(
        STTS_REWARD_CORRECT_WEIGHT * correctness
        + STTS_REWARD_PERF_WEIGHT * reward_speedup,
        6,
    )


def drkernel_turn_metrics(eval_result: dict[str, Any]) -> dict[str, str]:
    metadata = eval_result.get("metadata") or {}
    compiled = _first_present(
        eval_result.get("compiled"), eval_result.get("compilation")
    )
    correctness = eval_result.get("correctness")
    speedup = _as_float(eval_result.get("speedup"))
    decoy = _first_present(
        eval_result.get("decoy_kernel"), eval_result.get("is_decoy_kernel")
    )
    custom_time = _as_float(
        _first_present(
            eval_result.get("custom_kernel_cuda_time_in_profiling_us"),
            metadata.get("custom_kernel_cuda_time_in_profiling_us"),
        )
    )
    total_time = _as_float(
        _first_present(
            eval_result.get("total_kernel_run_time_in_profiling_us"),
            metadata.get("total_kernel_run_time_in_profiling_us"),
        )
    )
    time_coverage = None
    if custom_time is not None and total_time:
        time_coverage = custom_time / total_time

    num_custom = _first_present(
        eval_result.get("num_custom_kernel"),
        eval_result.get("num_custom_kernels"),
        metadata.get("num_custom_kernel"),
        metadata.get("num_custom_kernels"),
    )
    num_total = _first_present(
        eval_result.get("num_total_kernels"),
        metadata.get("num_total_kernels"),
    )
    num_coverage = _first_present(
        eval_result.get("num_coverage"),
        metadata.get("num_coverage"),
        num_custom,
    )
    error = _first_present(
        eval_result.get("error_message"),
        eval_result.get("error"),
        (eval_result.get("metadata") or {}).get("error"),
    )
    speedup_positive = bool(speedup is not None and speedup > 1.0)
    success = bool(compiled is True and correctness is True)

    return {
        "correctness": _metric_str(correctness),
        "performance": _metric_str(speedup),
        "speedup": _metric_str(speedup),
        "is_speedup_positive": _metric_str(speedup_positive),
        "is_decoy_kernel": _metric_str(bool(decoy)),
        "decoy_kernel": _metric_str(bool(decoy)),
        "compilation": _metric_str(compiled),
        "compiled": _metric_str(compiled),
        "success": _metric_str(success),
        "status": _metric_str(eval_result.get("status")),
        "error": _metric_str(error),
        "num_custom_kernel": _metric_str(num_custom),
        "num_total_kernels": _metric_str(num_total),
        "num_coverage": _metric_str(num_coverage),
        "custom_kernel_cuda_time_in_profiling_us": _metric_str(custom_time),
        "total_kernel_run_time_in_profiling_us": _metric_str(total_time),
        "time_coverage": _metric_str(time_coverage),
        "correctness_tensor": f"tensor([{1.0 if correctness is True else 0.0:g}])",
        "performance_tensor": f"tensor([{speedup if speedup is not None else 0.0:g}])",
        "compilation_tensor": f"tensor([{1.0 if compiled is True else 0.0:g}])",
    }


def drkernel_conversation_row(
    item: WorkItem,
    row: dict[str, Any],
    *,
    global_turn_idx: int,
) -> dict[str, Any]:
    score = float(row.get("score") or 0.0)
    row_turns = row.get("turns") or [
        {
            "turn_id": 1,
            "raw_response": row.get("raw_response") or "",
            "kernel_code": row.get("kernel_code") or "",
            "score": score,
            "metrics": row.get("metrics") or {},
        }
    ]
    turns = []
    for offset, turn in enumerate(row_turns):
        turns.append(
            {
                "turn_id": int(turn.get("turn_id") or (offset + 1)),
                "response": response_text_from_turn(turn),
                "score": float(turn.get("score") or 0.0),
                "global_turn_idx": global_turn_idx + offset,
                "metrics": turn.get("metrics") or {},
            }
        )
    return {
        "uid": f"claude_opus4_problem_{item.problem_id}_sample_{item.sample_id}",
        "num_turns": len(turns),
        "total_score": score,
        "turns": turns,
    }


def item_from_result_row(row: dict[str, Any]) -> WorkItem:
    return WorkItem(
        row_index=int(row.get("row_index", 0)),
        problem_id=int(row["problem_id"]),
        name=str(row.get("name", "")),
        prompt_text=strip_legacy_role_prefix(str(row.get("prompt", ""))),
        reference_code=str(row.get("reference_code", "")),
        sample_id=int(row.get("sample_id", 0)),
    )


def default_output_dir(dataset: Path, model: str) -> Path:
    stem = dataset.stem
    suffix = "mini10" if stem.endswith("_mini10") else stem
    return DEFAULT_EVAL_ROOT / f"claude-{model}-{suffix}"


class ResultStore:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.results_jsonl = output_dir / "results.jsonl"
        self.results_json = output_dir / "results.json"
        self.summary_json = output_dir / "summary.json"
        self.raw_responses_jsonl = output_dir / "raw_responses.jsonl"
        self.graded_conversations_jsonl = (
            output_dir / "graded_results_conversations_conversations.jsonl"
        )
        self.metrics_json = output_dir / "metrics.json"
        self.graded_results_parquet = output_dir / "graded_results.parquet"
        self._lock = threading.Lock()
        self._rows_by_key = self._load_existing()

    def _load_existing(self) -> dict[str, dict[str, Any]]:
        if not self.results_jsonl.exists():
            return {}
        rows: dict[str, dict[str, Any]] = {}
        for line in self.results_jsonl.read_text().splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = str(row.get("key") or "")
            if key:
                rows[key] = row
        return rows

    def has_completed(self, key: str, requested_turns: int = 1) -> bool:
        row = self._rows_by_key.get(key)
        return bool(
            row
            and row.get("stage") == "evaluated"
            and int(row.get("num_turns") or 1) >= requested_turns
        )

    def write(self, row: dict[str, Any]) -> None:
        row = json_safe(row)
        key = str(row["key"])
        with self._lock:
            self._rows_by_key[key] = row
            rows = sorted(
                self._rows_by_key.values(),
                key=lambda r: (
                    int(r.get("problem_id", -1)),
                    int(r.get("sample_id", -1)),
                ),
            )
            self.results_jsonl.write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                encoding="utf-8",
            )
            self.results_json.write_text(
                json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            self.summary_json.write_text(
                json.dumps(summarize(rows), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            self._write_drkernel_compatible_files(rows)

    def _write_drkernel_compatible_files(self, rows: list[dict[str, Any]]) -> None:
        evaluated_rows = [row for row in rows if row.get("stage") == "evaluated"]
        raw_lines = []
        conversation_lines = []
        parquet_rows = []
        for turn_idx, row in enumerate(evaluated_rows):
            item = item_from_result_row(row)
            uid = f"claude_opus4_problem_{item.problem_id}_sample_{item.sample_id}"
            raw_lines.append(
                {
                    "uid": uid,
                    "problem_id": item.problem_id,
                    "sample_id": item.sample_id,
                    "input": item.prompt_text,
                    "output": row.get("raw_response") or "",
                }
            )
            conversation = drkernel_conversation_row(
                item, row, global_turn_idx=turn_idx
            )
            conversation_lines.append(conversation)
            per_turn_scores = [
                float(turn.get("score") or 0.0) for turn in row.get("turns") or []
            ]
            parquet_rows.append(
                {
                    "uid": uid,
                    "problem_id": item.problem_id,
                    "sample_id": item.sample_id,
                    "score": row.get("score"),
                    "num_turns": row.get("num_turns"),
                    "per_turn_scores_json": json.dumps(
                        per_turn_scores, ensure_ascii=False
                    ),
                    "compiled": row.get("compiled"),
                    "correctness": row.get("correctness"),
                    "speedup": row.get("speedup"),
                    "turns_json": json.dumps(conversation["turns"], ensure_ascii=False),
                    "metrics_json": json.dumps(
                        row.get("metrics") or {}, ensure_ascii=False
                    ),
                }
            )

        self.raw_responses_jsonl.write_text(
            "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in raw_lines),
            encoding="utf-8",
        )
        self.graded_conversations_jsonl.write_text(
            "".join(
                json.dumps(line, ensure_ascii=False) + "\n"
                for line in conversation_lines
            ),
            encoding="utf-8",
        )
        self.metrics_json.write_text(
            json.dumps(drkernel_metrics_json(rows), ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
        try:
            pd.DataFrame(parquet_rows).to_parquet(
                self.graded_results_parquet, index=False
            )
        except Exception as exc:
            (self.output_dir / "graded_results.parquet.error.txt").write_text(
                repr(exc) + "\n",
                encoding="utf-8",
            )


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    evaluated = [r for r in rows if r.get("stage") == "evaluated"]
    compiled = [r for r in evaluated if r.get("compiled") is True]
    correct = [r for r in evaluated if r.get("correctness") is True]
    speedups = [float(r["speedup"]) for r in correct if r.get("speedup") is not None]
    solved = [v for v in speedups if v >= 1.0]
    return {
        "total_rows": len(rows),
        "evaluated": len(evaluated),
        "compiled": len(compiled),
        "correct": len(correct),
        "speedup_ge_1": len(solved),
        "best_speedup": max(speedups) if speedups else None,
        "mean_correct_speedup": (sum(speedups) / len(speedups)) if speedups else None,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }


def _rate(values: list[bool]) -> float | None:
    if not values:
        return None
    return sum(1 for value in values if value) / len(values)


def drkernel_metrics_json(rows: list[dict[str, Any]]) -> dict[str, Any]:
    evaluated = [r for r in rows if r.get("stage") == "evaluated"]
    scores = [float(r.get("score") or 0.0) for r in evaluated]
    speedups = [
        float(r["speedup"])
        for r in evaluated
        if r.get("speedup") is not None and r.get("correctness") is True
    ]
    compiled = [r.get("compiled") is True for r in evaluated]
    correct = [r.get("correctness") is True for r in evaluated]
    speedup_ge_1 = [
        r.get("correctness") is True
        and _as_float(r.get("speedup")) is not None
        and float(r["speedup"]) >= 1.0
        for r in evaluated
    ]
    mean_score = (sum(scores) / len(scores)) if scores else None
    metrics: dict[str, Any] = {
        "val/test_score/kernelbench_level2_validation": mean_score,
        "val/test_score/kernelbench_level2_validation_pass@1": _rate(correct),
        "val/test_score_extra/kernelbench_level2_validation/mean_score": mean_score,
        "val/multiturn/num_turns/mean": (
            (sum(int(r.get("num_turns") or 1) for r in evaluated) / len(evaluated))
            if evaluated
            else None
        ),
        "val/kernel/turn_1/compilation_rate": _rate(compiled),
        "val/kernel/turn_1/correctness_rate": _rate(correct),
        "val/kernel/turn_1/speedup_ge_1_rate": _rate(speedup_ge_1),
        "val/kernel/turn_1/mean_correct_speedup": (
            (sum(speedups) / len(speedups)) if speedups else None
        ),
        "val/kernel/turn_1/best_speedup": max(speedups) if speedups else None,
        "num_rows": len(rows),
        "num_evaluated": len(evaluated),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    max_turns = max((int(row.get("num_turns") or 1) for row in evaluated), default=0)
    for turn_id in range(1, max_turns + 1):
        turn_rows = []
        for row in evaluated:
            turns = row.get("turns") or []
            if len(turns) >= turn_id:
                turn_rows.append(turns[turn_id - 1])
        turn_compiled = [turn.get("compiled") is True for turn in turn_rows]
        turn_correct = [turn.get("correctness") is True for turn in turn_rows]
        turn_speedup_ge_1 = [
            turn.get("correctness") is True
            and _as_float(turn.get("speedup")) is not None
            and float(turn["speedup"]) >= 1.0
            for turn in turn_rows
        ]
        turn_speedups = [
            float(turn["speedup"])
            for turn in turn_rows
            if turn.get("speedup") is not None and turn.get("correctness") is True
        ]
        metrics[f"val/kernel/turn_{turn_id}/compilation_rate"] = _rate(turn_compiled)
        metrics[f"val/kernel/turn_{turn_id}/correctness_rate"] = _rate(turn_correct)
        metrics[f"val/kernel/turn_{turn_id}/speedup_ge_1_rate"] = _rate(
            turn_speedup_ge_1
        )
        metrics[f"val/kernel/turn_{turn_id}/mean_correct_speedup"] = (
            (sum(turn_speedups) / len(turn_speedups)) if turn_speedups else None
        )
        metrics[f"val/kernel/turn_{turn_id}/best_speedup"] = (
            max(turn_speedups) if turn_speedups else None
        )
    return metrics


_FEEDBACK_ERROR_HEAD = 3000
_FEEDBACK_ERROR_TAIL = 1000


def compact_eval_feedback(eval_result: dict[str, Any]) -> dict[str, Any]:
    """Distil the server eval_result to the fields the model actually needs.

    The raw payload is ~8KB/turn (full-precision profiling table, nested aten
    alias rows, task ids, hardware strings); this keeps ~1KB of signal. The
    full eval_result is still stored in results.jsonl — only the prompt shown
    to the model is compacted.
    """
    metadata = eval_result.get("metadata") or {}
    out: dict[str, Any] = {}
    for key in (
        "status",
        "compiled",
        "correctness",
        "speedup",
        "reference_runtime",
        "kernel_runtime",
        "error_code",
    ):
        value = eval_result.get(key)
        if value is not None:
            out[key] = value
    error_message = eval_result.get("error_message")
    if error_message:
        text = str(error_message)
        if len(text) > _FEEDBACK_ERROR_HEAD + _FEEDBACK_ERROR_TAIL:
            text = (
                text[:_FEEDBACK_ERROR_HEAD]
                + "\n...[truncated]...\n"
                + text[-_FEEDBACK_ERROR_TAIL:]
            )
        out["error_message"] = text
    for key in (
        "correctness_trials",
        "num_custom_kernels",
        "num_total_kernels",
        "custom_kernel_cuda_time_in_profiling_us",
        "total_kernel_run_time_in_profiling_us",
    ):
        value = metadata.get(key)
        if value is not None:
            out[key] = value
    kernels = (metadata.get("profiling") or {}).get("kernels") or []
    rows: list[str] = []
    seen_times: set[int] = set()
    for entry in kernels:
        cuda_us = round(float(entry.get("cuda_time_us") or 0.0))
        name = str(entry.get("name") or "")
        # nested aten:: wrappers report the same cuda time as their parent op;
        # keep only the first row per distinct timing
        if cuda_us in seen_times and name.startswith("aten::"):
            continue
        seen_times.add(cuda_us)
        if len(name) > 120:
            name = name[:120] + "…"
        rows.append(f"{name}: {cuda_us}us x{entry.get('count')}")
        if len(rows) >= 8:
            break
    if rows:
        out["profiling_top_kernels"] = rows
    return out


def feedback_prompt(
    eval_result: dict[str, Any], ncu_report: str | None = None
) -> str:
    compact = compact_eval_feedback(eval_result)
    ncu_section = ""
    if ncu_report:
        ncu_section = f"""
NCU expert analysis (Nsight Compute hierarchical profiling of your kernel):
{ncu_report.rstrip()}
"""
    return f"""Now you have received the server feedback for your last implementation. Based on that and all your previous responses, improve the implementation.

Here is the server feedback. Please refer to this feedback to improve the implementation:
Server feedback (status/metrics/errors):
{json.dumps(json_safe(compact), ensure_ascii=False, indent=1)}
{ncu_section}
Return an improved Triton implementation named `ModelNew` as a single ```python``` block. Before the code block, include a concise visible `Optimization notes:` section. Do not include hidden chain-of-thought.
"""


def response_text_from_turn(turn: dict[str, Any]) -> str:
    response = str(turn.get("raw_response") or "")
    if response:
        return response.rstrip()
    kernel_code = str(turn.get("kernel_code") or "")
    return f"```python\n{kernel_code.rstrip()}\n```"


def conversation_text(item: WorkItem, turns: list[dict[str, Any]]) -> str:
    parts = ["[user]", strip_legacy_role_prefix(item.prompt_text).rstrip()]
    for turn in turns:
        parts.extend(["", "[assistant]", response_text_from_turn(turn)])
        if turn.get("eval_result") is not None:
            parts.extend(
                [
                    "",
                    "[user]",
                    feedback_prompt(
                        turn["eval_result"], ncu_report=turn.get("ncu_report")
                    ).rstrip(),
                ]
            )
    return "\n".join(parts).rstrip() + "\n"


def _turn_reward(t: dict[str, Any]) -> float:
    """STTS selection key: drkernel reward if present, else fall back to score."""
    r = t.get("reward")
    if r is not None:
        return float(r)
    return float(t.get("score") or 0.0)


def select_stts_history(
    turns: list[dict[str, Any]], top_k: int
) -> list[dict[str, Any]]:
    """Pick the top_k highest-REWARD turns (drkernel best-K; ties favour later
    turns), returned in chronological order for STTS prompt construction."""
    ranked = sorted(
        turns,
        key=lambda t: (_turn_reward(t), int(t.get("turn_id") or 0)),
        reverse=True,
    )[:top_k]
    return sorted(ranked, key=lambda t: int(t.get("turn_id") or 0))


def stts_note_text(
    *,
    turn_id: int,
    total_turns: int,
    all_turns: list[dict[str, Any]],
    selected: list[dict[str, Any]],
) -> str:
    best_reward = max((_turn_reward(t) for t in all_turns), default=0.0)
    best_speedup = max((float(t.get("score") or 0.0) for t in all_turns), default=0.0)
    shown = ", ".join(str(t.get("turn_id")) for t in selected)
    return (
        f"Note: this is turn {turn_id} of {total_turns}. You have made "
        f"{len(all_turns)} previous attempts; only your {len(selected)} "
        f"best-reward attempts (turns {shown}; reward = 0.5*correct + "
        f"0.5*min(speedup,3.0), i.e. correctness matters and speedup is capped "
        f"at 3x) are shown above, in chronological order. Best reward so far "
        f"{best_reward:.4f} (best speedup {best_speedup:.4f}). Analyze why the "
        "best attempts performed well and produce a new implementation that "
        "beats them."
    )


def build_claude_prompt(
    item: WorkItem,
    *,
    turn_id: int = 1,
    previous_turn: dict[str, Any] | None = None,
    history: list[dict[str, Any]] | None = None,
    total_turns: int = 1,
    stts_note: str | None = None,
) -> str:
    if turn_id <= 1:
        return prompt_with_visible_notes_instruction(item.prompt_text) + "\n"

    turns = list(history or [])
    if not turns and previous_turn:
        turns = [previous_turn]
    if not turns:
        raise ValueError("later turns require previous turn feedback")
    text = conversation_text(item, turns)
    if stts_note:
        text = text.rstrip() + "\n\n" + stts_note.rstrip() + "\n"
    return text


def write_turn_optimization_notes(
    item: WorkItem,
    *,
    output_dir: Path,
    turn_id: int,
    raw_response: str,
) -> tuple[str, Path]:
    notes_dir = output_dir / "reasoning_summaries"
    notes_dir.mkdir(parents=True, exist_ok=True)
    notes = extract_optimization_notes(raw_response)
    notes_path = notes_dir / f"{item.key}_turn_{turn_id}.txt"
    notes_path.write_text(notes.rstrip() + "\n", encoding="utf-8")
    return notes, notes_path


# Run `claude -p` from a stable directory outside the repo so it does not load
# the repo's CLAUDE.md, project memory, or MCP servers — none are needed for
# kernel generation and together they add ~7K input tokens per call.
CLAUDE_CLEAN_CWD = Path(os.getenv("KG_CLAUDE_CWD", "/tmp/kernelgym_claude_cwd"))


def _parse_claude_stdout(stdout: str) -> tuple[str, dict[str, Any]]:
    """Parse the `claude -p --output-format json` envelope into (response_text,
    usage_info). Falls back to treating stdout as plain text when it is not the
    expected envelope (e.g. older CLI or plain-text mode)."""
    try:
        envelope = json.loads(stdout)
    except (TypeError, ValueError):
        return stdout, {}
    if not isinstance(envelope, dict) or "result" not in envelope:
        return stdout, {}
    info = {
        "usage": envelope.get("usage"),
        "cost_usd": envelope.get("total_cost_usd"),
        "session_id": envelope.get("session_id"),
        "duration_api_ms": envelope.get("duration_api_ms"),
        "is_error": envelope.get("is_error"),
    }
    return str(envelope.get("result") or ""), {
        k: v for k, v in info.items() if v is not None
    }


# --- skill-driven hierarchical NCU analysis ---------------------------------
# A second, cheaper `claude -p` call diagnoses the kernel from NCU metrics
# using the repo's GPU-profiling skills, and decides which metrics the server
# should collect on the NEXT turn (LLM-directed tier escalation).
ANALYSIS_CWD = Path(
    os.getenv("KG_ANALYSIS_CWD", "/tmp/kernelgym_claude_analysis_cwd")
)
ANALYSIS_SKILLS = (
    "gpu-kernel-diag",
    "gpu-kernel-analyzer",
    "kernel-opt-strategy",
)


def setup_analysis_cwd() -> Path:
    """Create the analysis working dir with .claude/skills symlinks so the
    analysis `claude -p` call discovers the GPU-profiling skills (the dir
    lives outside the repo so the repo CLAUDE.md / MCP servers do not load)."""
    skills_dir = ANALYSIS_CWD / ".claude" / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)
    for name in ANALYSIS_SKILLS:
        link = skills_dir / name
        target = ROOT / "skills" / name
        if not link.exists() and target.is_dir():
            link.symlink_to(target)
    return ANALYSIS_CWD


def extract_last_json_block(text: str) -> dict[str, Any] | None:
    blocks = re.findall(r"```json\s*(.*?)```", text, flags=re.DOTALL)
    for raw in reversed(blocks):
        try:
            parsed = json.loads(raw)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def format_ncu_for_analysis(
    ncu_payload: dict[str, Any], *, top_k: int = 10
) -> dict[str, Any]:
    """Regroup the server NCU payload (metric -> per-launch list) into
    per-kernel averages ranked by total GPU time, which is what the diagnosis
    skill expects to read."""
    metrics = (ncu_payload or {}).get("metrics") or {}
    sums: dict[str, dict[str, list[float]]] = {}
    for mname, m in metrics.items():
        for pk in m.get("per_kernel") or []:
            value = pk.get("value")
            if value is None:
                continue
            kname = str(pk.get("kernel_name") or "?").split("(")[0][:100]
            sums.setdefault(kname, {}).setdefault(mname, []).append(float(value))
    per_kernel: dict[str, dict[str, Any]] = {}
    for kname, mvals in sums.items():
        row = {m: round(sum(v) / len(v), 4) for m, v in mvals.items()}
        row["__launch_count"] = max(len(v) for v in mvals.values())
        per_kernel[kname] = row
    ranked = sorted(
        per_kernel.items(),
        key=lambda kv: -(
            kv[1].get("gpu__time_duration.sum", 0.0) * kv[1]["__launch_count"]
        ),
    )[:top_k]
    return {
        "note": "values are per-launch averages; kernels ranked by total GPU time",
        "ncu_status": (ncu_payload or {}).get("status"),
        "kernels": dict(ranked),
    }


def build_analysis_prompt(
    item: WorkItem,
    *,
    turn_id: int,
    kernel_code: str,
    eval_result: dict[str, Any],
    ncu_payload: dict[str, Any],
) -> str:
    compact = compact_eval_feedback(eval_result)
    pack_names = ", ".join(PACKS.keys())
    return f"""Use the Skill tool to invoke the `gpu-kernel-diag` skill, then diagnose the GPU kernel below in dialog-diagnosis mode (you CANNOT run commands; all profiling data is provided here). Consult `gpu-kernel-analyzer` if you need the decision tree.

## Problem
{item.name} (KernelBench, Triton kernel vs torch_compile reference)

## Candidate kernel (turn {turn_id})
```python
{kernel_code[:12000]}
```

## Timing / correctness (server eval)
```json
{json.dumps(json_safe(compact), ensure_ascii=False, indent=1)}
```

## NCU metrics (hierarchical collection, per-kernel averages)
```json
{json.dumps(json_safe(format_ncu_for_analysis(ncu_payload)), ensure_ascii=False, indent=1)[:20000]}
```

## Required output (strict)
1. `## Diagnosis` — bottleneck + evidence, <=250 words. Cite metric values.
2. `## Fixes` — top-3 concrete actions, each mapped to Triton meta-parameters or code-structure changes (diag_rules.md rule 10).
3. A final ```json block deciding NEXT-turn NCU collection (metric_tiers.md §4):
```json
{{"packs": ["PACK-..."], "next_metrics": [], "reason": "...", "expected_discrimination": "..."}}
```
Allowed packs: {pack_names}. `next_metrics` may only contain metric names that appear in the data above or in metric_tiers.md tier lists — never invent names. Choose the SMALLEST set that discriminates your remaining hypotheses; empty packs+metrics means T0 screening only.
"""


def run_skill_analysis(
    item: WorkItem,
    *,
    output_dir: Path,
    model: str,
    effort: str,
    timeout_sec: int,
    turn_id: int,
    kernel_code: str,
    eval_result: dict[str, Any],
    ncu_payload: dict[str, Any],
) -> tuple[str, list[str], list[str], dict[str, Any]]:
    """Returns (report_markdown, requested_metrics, requested_packs, meta)."""
    setup_analysis_cwd()
    prompt = build_analysis_prompt(
        item,
        turn_id=turn_id,
        kernel_code=kernel_code,
        eval_result=eval_result,
        ncu_payload=ncu_payload,
    )
    analysis_dir = output_dir / "ncu_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / f"{item.key}_turn_{turn_id}_metrics.json").write_text(
        json.dumps(json_safe(ncu_payload), ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    cmd = [
        "claude",
        "-p",
        "--model",
        model,
        "--effort",
        effort,
        "--no-session-persistence",
        "--dangerously-skip-permissions",
        "--output-format",
        "json",
        "--strict-mcp-config",
        "--mcp-config",
        '{"mcpServers":{}}',
        "--disallowedTools",
        "Bash,Edit,Write,NotebookEdit,WebFetch,WebSearch",
    ]
    completed = subprocess.run(
        cmd,
        input=prompt,
        text=True,
        capture_output=True,
        timeout=timeout_sec,
        check=False,
        cwd=setup_analysis_cwd(),
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"analysis claude -p rc={completed.returncode}: "
            f"{(completed.stderr or completed.stdout or '').strip()[-500:]!r}"
        )
    report, info = _parse_claude_stdout(completed.stdout)
    decision = extract_last_json_block(report) or {}
    packs = [str(p) for p in (decision.get("packs") or [])]
    next_metrics = [str(m) for m in (decision.get("next_metrics") or [])]
    (analysis_dir / f"{item.key}_turn_{turn_id}_report.md").write_text(
        report, encoding="utf-8"
    )
    meta = {f"analysis_{k}": v for k, v in info.items()}
    return report, next_metrics, packs, meta


# --- two-stage strategy analysis (nsys/explore + ncu/exploit) ---------------
def format_nsys_for_analysis(nsys_payload: dict[str, Any]) -> dict[str, Any]:
    """Project the server nsys payload into the timeline signals the strategy
    skill reads: kernel count, hotspots, launch/sync/memcpy overhead."""
    s = (nsys_payload or {}).get("summary") or {}
    return {
        "nsys_status": (nsys_payload or {}).get("status"),
        "num_distinct_kernels": s.get("num_distinct_kernels"),
        "total_kernel_time_ns": s.get("total_kernel_time_ns"),
        "cuda_launch_api_ns": s.get("cuda_launch_api_ns"),
        "cuda_sync_api_ns": s.get("cuda_sync_api_ns"),
        "total_memops_time_ns": s.get("total_memops_time_ns"),
        "top_kernels": (s.get("top_kernels") or [])[:10],
        "mem_ops": (s.get("mem_ops") or [])[:6],
    }


def build_strategy_prompt(
    item: WorkItem,
    *,
    turn_id: int,
    phase: str,
    best_speedup: float,
    kernel_code: str,
    eval_result: dict[str, Any],
    nsys_payload: dict[str, Any],
    ncu_payload: dict[str, Any] | None,
) -> str:
    compact = compact_eval_feedback(eval_result)
    pack_names = ", ".join(PACKS.keys())
    nsys_block = json.dumps(
        json_safe(format_nsys_for_analysis(nsys_payload)),
        ensure_ascii=False,
        indent=1,
    )[:8000]
    if phase == "exploit":
        skill_line = (
            "Use the Skill tool to invoke `kernel-opt-strategy` (confirm we are "
            "in EXPLOIT phase), then invoke `gpu-kernel-diag` to do NCU local "
            "tuning of the hotspot kernel. Consult `gpu-kernel-analyzer` if a "
            "metric signal is ambiguous."
        )
        ncu_section = (
            "## NCU metrics (hotspot kernel internals, per-kernel averages)\n"
            "```json\n"
            + json.dumps(
                json_safe(format_ncu_for_analysis(ncu_payload or {})),
                ensure_ascii=False,
                indent=1,
            )[:16000]
            + "\n```\n"
        )
        output_spec = (
            "1. `## Diagnosis` — the hotspot kernel's internal bottleneck, "
            "<=250 words, cite NCU + nsys values.\n"
            "2. `## Fixes` — top-3 LOCAL tuning actions (tile/BLOCK/num_warps/"
            "num_stages/occupancy/coalescing/precision/eviction), each mapped "
            "to a concrete Triton meta-parameter. Do NOT restructure the kernel "
            "set unless the strategy skill's exploration-revival rule fires.\n"
            "3. Final ```json deciding phase + next-turn NCU collection."
        )
    else:
        skill_line = (
            "Use the Skill tool to invoke `kernel-opt-strategy`, then work in "
            "EXPLORE phase: judge whether the current kernel DECOMPOSITION is "
            "right before tuning any single kernel. Use the nsys timeline "
            "(kernel count / launch+memcpy overhead / hotspot) to decide what "
            "to FUSE, ELIMINATE, or replace — consult methods.md for the 7 "
            "high-leverage techniques (math-equivalence, folding, fusion)."
        )
        ncu_section = ""
        output_spec = (
            "1. `## Diagnosis` — is the current kernel set structurally right? "
            "How many kernels, where is glue/round-trip waste (cite nsys: "
            "num_distinct_kernels, memops, launch overhead, hotspot %)? <=250 "
            "words.\n"
            "2. `## Fixes` — top-3 STRUCTURAL actions: which methods.md "
            "technique to apply (fuse these kernels / fold this op / math-"
            "simplify this / eliminate this round-trip). Push for the biggest "
            "algorithmic lever, not micro-tuning.\n"
            "3. Final ```json deciding phase."
        )
    return f"""{skill_line}

## Problem
{item.name} (KernelBench, Triton kernel vs torch_compile reference)

## Phase / progress
current_phase={phase}; turn={turn_id}; best_speedup_so_far={best_speedup:.4f}

## Candidate kernel (turn {turn_id})
```python
{kernel_code[:12000]}
```

## Timing / correctness (server eval)
```json
{json.dumps(json_safe(compact), ensure_ascii=False, indent=1)}
```

## NSYS timeline (system / inter-kernel view — kernel count, hotspots, overhead)
```json
{nsys_block}
```
{ncu_section}
## Required output (strict)
{output_spec}
```json
{{"phase": "explore|exploit", "phase_reason": "...", "untried_structural_levers": [], "packs": [], "next_metrics": []}}
```
Phase rule (kernel-opt-strategy): stay `explore` while structure can still be improved or best_speedup keeps jumping >15%.
`untried_structural_levers` = list any NOT-yet-tried **method/structural** action your own Fixes propose — INCLUDING intra-kernel GEMM/Conv formulation swaps (`tl.dot` implicit-GEMM ↔ direct-FMA accumulation ↔ library call), operator folding, fusion, algebraic simplification. Tile/occupancy/precision micro-tuning does NOT count.
🛑 You may set `phase: "exploit"` ONLY when `untried_structural_levers` is EMPTY **and** >=2 turns gained <5% **and** the bottleneck is pure implementation detail. If that list is non-empty, you MUST stay `explore` and the next turn must execute the top lever — intra-kernel algorithm choice (e.g. tl.dot vs FMA for a pathological K/N shape) is STRUCTURAL, not a micro-tune. (Lesson P65: a padded `tl.dot` on K=72/N=64 wasted 44% MACs; switching to it via exploit-micro-tuning capped at 1.73x while direct-FMA reaches ~5x.)
In `exploit`, `packs` may be: {pack_names}; `next_metrics` only names present in the data above. Empty = T0 screening only. If exploitation stalls at roofline, set phase back to `explore`.
"""


def run_strategy_analysis(
    item: WorkItem,
    *,
    output_dir: Path,
    model: str,
    effort: str,
    timeout_sec: int,
    turn_id: int,
    phase: str,
    best_speedup: float,
    kernel_code: str,
    eval_result: dict[str, Any],
    nsys_payload: dict[str, Any],
    ncu_payload: dict[str, Any] | None,
) -> tuple[str, str, list[str], list[str], dict[str, Any]]:
    """Returns (report, next_phase, requested_metrics, requested_packs, meta)."""
    setup_analysis_cwd()
    prompt = build_strategy_prompt(
        item,
        turn_id=turn_id,
        phase=phase,
        best_speedup=best_speedup,
        kernel_code=kernel_code,
        eval_result=eval_result,
        nsys_payload=nsys_payload,
        ncu_payload=ncu_payload,
    )
    analysis_dir = output_dir / "strategy_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / f"{item.key}_turn_{turn_id}_profiling.json").write_text(
        json.dumps(
            json_safe(
                {
                    "nsys": format_nsys_for_analysis(nsys_payload),
                    "ncu": format_ncu_for_analysis(ncu_payload or {})
                    if ncu_payload
                    else None,
                }
            ),
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    cmd = [
        "claude",
        "-p",
        "--model",
        model,
        "--effort",
        effort,
        "--no-session-persistence",
        "--dangerously-skip-permissions",
        "--output-format",
        "json",
        "--strict-mcp-config",
        "--mcp-config",
        '{"mcpServers":{}}',
        "--disallowedTools",
        "Bash,Edit,Write,NotebookEdit,WebFetch,WebSearch",
    ]
    completed = subprocess.run(
        cmd,
        input=prompt,
        text=True,
        capture_output=True,
        timeout=timeout_sec,
        check=False,
        cwd=setup_analysis_cwd(),
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"strategy analysis claude -p rc={completed.returncode}: "
            f"{(completed.stderr or completed.stdout or '').strip()[-500:]!r}"
        )
    report, info = _parse_claude_stdout(completed.stdout)
    decision = extract_last_json_block(report) or {}
    next_phase = str(decision.get("phase") or phase).strip().lower()
    if next_phase not in {"explore", "exploit"}:
        next_phase = phase
    # Guardrail (P65 lesson): never let the run leave EXPLORE while the analysis
    # itself still lists untried method/structural levers (incl. intra-kernel
    # GEMM-formulation swaps tl.dot<->FMA<->lib). Premature exploit micro-tunes
    # the wrong kernel form and caps speedup.
    untried = [str(x) for x in (decision.get("untried_structural_levers") or []) if str(x).strip()]
    forced = False
    if untried and next_phase == "exploit":
        next_phase = "explore"
        forced = True
    packs = [str(p) for p in (decision.get("packs") or [])]
    next_metrics = [str(m) for m in (decision.get("next_metrics") or [])]
    (analysis_dir / f"{item.key}_turn_{turn_id}_report.md").write_text(
        report, encoding="utf-8"
    )
    meta = {f"analysis_{k}": v for k, v in info.items()}
    meta["analysis_phase"] = phase
    meta["analysis_next_phase"] = next_phase
    meta["analysis_untried_structural_levers"] = untried
    if forced:
        meta["analysis_phase_forced_explore"] = True
        print(
            f"[strategy] {item.key} turn {turn_id}: forced explore (untried "
            f"structural levers: {untried})",
            flush=True,
        )
    return report, next_phase, next_metrics, packs, meta


QUOTA_WAIT_SECONDS = int(os.getenv("CLAUDE_QUOTA_WAIT_SECONDS", "300"))
TRANSIENT_RETRIES = int(os.getenv("CLAUDE_TRANSIENT_RETRIES", "5"))
TRANSIENT_WAIT_SECONDS = int(os.getenv("CLAUDE_TRANSIENT_WAIT_SECONDS", "30"))
_QUOTA_PATTERNS = (
    "usage limit",
    "rate limit",
    "rate_limit",
    "ratelimit",
    "quota",
    "overloaded",
    "too many requests",
    "429",
    "resource_exhausted",
    "please try again later",
    "capacity",
    "529",
)


def _looks_like_quota_error(completed: subprocess.CompletedProcess) -> bool:
    blob = f"{completed.stderr or ''}\n{completed.stdout or ''}".lower()
    return any(pat in blob for pat in _QUOTA_PATTERNS)


def _run_claude_with_quota_wait(
    cmd: list[str],
    *,
    prompt: str,
    timeout_sec: int,
    item_key: str,
    turn_id: int,
) -> subprocess.CompletedProcess:
    """Run `claude -p`, sleeping and retrying when quota/rate limits are hit.

    Retries indefinitely on quota-like failures so the run survives credit
    exhaustion and resumes automatically once quota recovers.
    """
    CLAUDE_CLEAN_CWD.mkdir(parents=True, exist_ok=True)
    attempt = 0
    wait = TRANSIENT_WAIT_SECONDS
    while True:
        attempt += 1
        try:
            completed = subprocess.run(
                cmd,
                input=prompt,
                text=True,
                capture_output=True,
                timeout=timeout_sec,
                check=False,
                cwd=CLAUDE_CLEAN_CWD,
            )
        except subprocess.TimeoutExpired:
            # claude -p stalled: treat identically to any other failure —
            # wait and retry forever so the run never needs human intervention.
            print(
                f"[retry] {item_key} turn {turn_id} timeout>{timeout_sec}s "
                f"(attempt {attempt}); retrying after {wait}s.",
                flush=True,
            )
            time.sleep(wait)
            wait = min(wait * 2, QUOTA_WAIT_SECONDS)
            continue
        if completed.returncode == 0:
            return completed
        tail = (
            f"stderr={(completed.stderr or '').strip()[-200:]!r} "
            f"stdout={(completed.stdout or '').strip()[-200:]!r}"
        )
        # All non-zero exits (quota, rate-limit, ECONNRESET, transient 5xx,
        # empty-stderr rc=1 …) are retried forever with exponential back-off
        # capped at QUOTA_WAIT_SECONDS.  The run self-heals without any human
        # interaction regardless of which error flavour claude CLI emits.
        if _looks_like_quota_error(completed):
            wait_this = QUOTA_WAIT_SECONDS  # quota: use long fixed interval
        else:
            wait_this = wait
            wait = min(wait * 2, QUOTA_WAIT_SECONDS)
        print(
            f"[retry] {item_key} turn {turn_id} rc={completed.returncode} "
            f"(attempt {attempt}); sleeping {wait_this}s. {tail}",
            flush=True,
        )
        time.sleep(wait_this)


def run_claude(
    item: WorkItem,
    *,
    output_dir: Path,
    model: str,
    timeout_sec: int,
    turn_id: int,
    prompt: str,
) -> tuple[str, dict[str, Any]]:
    prompts_dir = output_dir / "prompts"
    raw_dir = output_dir / "raw_responses"
    kernels_dir = output_dir / "kernels"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    kernels_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = prompts_dir / f"{item.key}_turn_{turn_id}.txt"
    raw_path = raw_dir / f"{item.key}_turn_{turn_id}.md"
    kernel_path = kernels_dir / f"{item.key}_turn_{turn_id}.py"

    prompt_path.write_text(prompt, encoding="utf-8")

    cmd = [
        "claude",
        "-p",
        "--model",
        model,
        "--effort",
        "max",
        "--no-session-persistence",
        "--dangerously-skip-permissions",
        "--output-format",
        "json",
        "--strict-mcp-config",
        "--mcp-config",
        '{"mcpServers":{}}',
        "--disallowedTools",
        "Bash,Edit,Write,Read,NotebookEdit,WebFetch,WebSearch",
    ]
    start = time.monotonic()
    completed = _run_claude_with_quota_wait(
        cmd, prompt=prompt, timeout_sec=timeout_sec, item_key=item.key, turn_id=turn_id
    )
    elapsed = time.monotonic() - start
    if completed.returncode != 0:
        # Persist full stdout+stderr so transient claude CLI failures (often
        # rc=1 with empty stderr and the real reason on stdout) are diagnosable.
        fail_dir = output_dir / "claude_failures"
        fail_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fail_path = fail_dir / f"{item.key}_turn_{turn_id}_{stamp}_rc{completed.returncode}.txt"
        fail_path.write_text(
            f"rc={completed.returncode}\n"
            f"elapsed_sec={round(elapsed, 3)}\n"
            f"===== STDOUT =====\n{completed.stdout or ''}\n"
            f"===== STDERR =====\n{completed.stderr or ''}\n",
            encoding="utf-8",
        )
        stdout_tail = (completed.stdout or "").strip()[-1000:]
        stderr_tail = (completed.stderr or "").strip()[-1000:]
        raise RuntimeError(
            f"claude -p failed with rc={completed.returncode}: "
            f"stderr={stderr_tail!r} stdout={stdout_tail!r} (saved {fail_path})"
        )
    raw_text, claude_info = _parse_claude_stdout(completed.stdout)
    raw_path.write_text(raw_text, encoding="utf-8")
    # Retry the whole claude call if the response contains no ModelNew block.
    # This happens when quota messages or truncated output land instead of code.
    _no_code_tries = 0
    while True:
        try:
            code = extract_python_code(raw_text)
            break
        except ValueError:
            _no_code_tries += 1
            if _no_code_tries > TRANSIENT_RETRIES:
                raise
            print(
                f"[no-code-retry] {item.key} turn {turn_id}: no ModelNew block "
                f"({_no_code_tries}/{TRANSIENT_RETRIES}); re-running claude. "
                f"response tail: {raw_text.strip()[-200:]!r}",
                flush=True,
            )
            time.sleep(TRANSIENT_WAIT_SECONDS)
            completed = _run_claude_with_quota_wait(
                cmd, prompt=prompt, timeout_sec=timeout_sec,
                item_key=item.key, turn_id=turn_id,
            )
            if completed.returncode != 0:
                stdout_tail = (completed.stdout or "").strip()[-1000:]
                stderr_tail = (completed.stderr or "").strip()[-1000:]
                raise RuntimeError(
                    f"claude -p failed with rc={completed.returncode}: "
                    f"stderr={stderr_tail!r} stdout={stdout_tail!r}"
                )
            raw_text, claude_info = _parse_claude_stdout(completed.stdout)
            raw_path.write_text(raw_text, encoding="utf-8")
    optimization_notes, optimization_notes_path = write_turn_optimization_notes(
        item,
        output_dir=output_dir,
        turn_id=turn_id,
        raw_response=raw_text,
    )
    kernel_path.write_text(code + "\n", encoding="utf-8")
    meta = {
        "raw_response": raw_text,
        "optimization_notes": optimization_notes,
        "optimization_notes_path": optimization_notes_path,
        "turn_id": turn_id,
        "claude_returncode": completed.returncode,
        "claude_elapsed_sec": round(elapsed, 3),
        "claude_stdout_tail": completed.stdout[-4000:],
        "claude_stderr_tail": completed.stderr[-4000:],
        "prompt_path": prompt_path,
        "raw_response_path": raw_path,
        "kernel_path": kernel_path,
        **{f"claude_{k}": v for k, v in claude_info.items()},
    }
    return code, meta


def evaluate_kernel(
    item: WorkItem,
    kernel_code: str,
    *,
    server_url: str,
    num_correct_trials: int,
    num_perf_trials: int,
    timeout_sec: int,
    enable_ncu_profiling: bool,
    ncu_metrics: list[str] | None = None,
    enable_nsys_profiling: bool = False,
) -> dict[str, Any]:
    task_id = f"claude-opus4-p{item.problem_id}-s{item.sample_id}-{uuid4().hex[:8]}"
    payload = {
        "task_id": task_id,
        "reference_code": item.reference_code,
        "kernel_code": kernel_code,
        "entry_point": "Model",
        "backend": "triton",
        "reference_backend": "torch_compile",
        "num_correct_trials": num_correct_trials,
        "num_perf_trials": num_perf_trials,
        "force_refresh": True,
        "enable_profiling": True,
        "enable_ncu_profiling": enable_ncu_profiling or bool(ncu_metrics),
        "enable_nsys_profiling": enable_nsys_profiling,
        "run_triton_detection": True,
    }
    if ncu_metrics:
        payload["ncu_metrics"] = list(ncu_metrics)
    timeout = httpx.Timeout(float(timeout_sec), connect=10.0)
    with httpx.Client(timeout=timeout) as client:
        response = client.post(f"{server_url.rstrip('/')}/evaluate", json=payload)
        response.raise_for_status()
        return response.json()


def write_eval_artifacts(
    item: WorkItem,
    *,
    output_dir: Path,
    turns: list[dict[str, Any]],
    total_score: float,
) -> dict[str, Path]:
    eval_dir = (
        output_dir
        / "eval_outputs"
        / f"problem_{item.problem_id}_sample_{item.sample_id}"
    )
    eval_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "eval_output_dir": eval_dir,
        "reference_path": eval_dir / "reference.py",
        "conversation_path": eval_dir / "full_conversation.txt",
        "optimization_notes_path": eval_dir / "optimization_notes.txt",
        "summary_path": eval_dir / "summary.json",
    }
    paths["reference_path"].write_text(
        item.reference_code.rstrip() + "\n", encoding="utf-8"
    )
    notes_dir = output_dir / "reasoning_summaries"
    notes_dir.mkdir(parents=True, exist_ok=True)
    notes_sections: list[str] = []
    for turn in turns:
        turn_id = int(turn["turn_id"])
        kernel_path = eval_dir / f"turn_{turn_id}_kernel.py"
        eval_path = eval_dir / f"turn_{turn_id}_eval.json"
        state_path = eval_dir / f"turn_{turn_id}_state.json"
        turn_notes_path = notes_dir / f"{item.key}_turn_{turn_id}.txt"
        notes = str(
            turn.get("optimization_notes")
            or extract_optimization_notes(str(turn.get("raw_response") or ""))
        )
        kernel_path.write_text(
            str(turn["kernel_code"]).rstrip() + "\n", encoding="utf-8"
        )
        eval_path.write_text(
            json.dumps(json_safe(turn["eval_result"]), ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
        state_path.write_text(
            json.dumps(json_safe(turn.get("state") or {}), ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
        turn_notes_path.write_text(notes.rstrip() + "\n", encoding="utf-8")
        notes_sections.append(f"[turn {turn_id}]\n{notes.rstrip()}")
        paths[f"turn_{turn_id}_kernel_path"] = kernel_path
        paths[f"turn_{turn_id}_eval_path"] = eval_path
        paths[f"turn_{turn_id}_state_path"] = state_path
        paths[f"turn_{turn_id}_optimization_notes_path"] = turn_notes_path
    notes_text = "\n\n".join(notes_sections).rstrip()
    paths["optimization_notes_path"].write_text(
        (notes_text + "\n") if notes_text else "", encoding="utf-8"
    )
    paths["conversation_path"].write_text(
        conversation_text(item, turns), encoding="utf-8"
    )
    per_turn_scores = [float(turn.get("score") or 0.0) for turn in turns]
    first_score = per_turn_scores[0] if per_turn_scores else 0.0
    last_score = per_turn_scores[-1] if per_turn_scores else 0.0
    paths["summary_path"].write_text(
        json.dumps(
            json_safe(
                {
                    "uid": f"claude_opus4_problem_{item.problem_id}_sample_{item.sample_id}",
                    "problem_id": item.problem_id,
                    "sample_id": item.sample_id,
                    "num_turns": len(turns),
                    "total_score": total_score,
                    "per_turn_scores": per_turn_scores,
                    "improvement": {
                        "first_turn_score": first_score,
                        "last_turn_score": last_score,
                        "improved": last_score > first_score,
                    },
                }
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return paths


def load_items(
    dataset: Path, sample_id: int, limit: int | None, problem_ids: set[int] | None
) -> list[WorkItem]:
    df = pd.read_parquet(dataset)
    items: list[WorkItem] = []
    for row_index, row in df.iterrows():
        extra = row["extra_info"]
        reward_model = row["reward_model"]
        problem_id = int(extra["problem_id"])
        if problem_ids and problem_id not in problem_ids:
            continue
        items.append(
            WorkItem(
                row_index=int(row_index),
                problem_id=problem_id,
                name=str(extra.get("name", "")),
                prompt_text=prompt_text_from_cell(row["prompt"]),
                reference_code=str(reward_model["ground_truth"]),
                sample_id=sample_id,
            )
        )
        if limit is not None and len(items) >= limit:
            break
    return items


def process_one(
    item: WorkItem,
    *,
    output_dir: Path,
    store: ResultStore,
    args: argparse.Namespace,
    eval_semaphore: threading.Semaphore,
) -> dict[str, Any]:
    start = time.monotonic()
    if args.resume and store.has_completed(item.key, requested_turns=args.num_turns):
        return {"key": item.key, "stage": "skipped"}

    claude_meta: dict[str, Any] = {}
    raw_text = ""
    turns: list[dict[str, Any]] = []
    use_skills = bool(getattr(args, "use_skills", False))
    use_strategy = bool(getattr(args, "use_strategy", False))
    analysis_effort = str(getattr(args, "analysis_effort", "high") or "high")
    analysis_model = str(
        getattr(args, "analysis_model", "") or args.model
    )
    # LLM-directed collection: first profiled turn grabs the FULL skill metric
    # space; afterwards the analysis agent's decision (T0 + chosen packs/
    # metrics) narrows the next collection. None => full space.
    pending_ncu_metrics: list[str] | None = None
    # two-stage strategy state: start in EXPLORE (structure/method search via
    # nsys), switch to EXPLOIT (NCU local tuning) when the method stabilizes.
    phase = "explore"
    best_speedup = 0.0
    try:
        for turn_id in range(1, args.num_turns + 1):
            history = turns
            stts_note = None
            stts_top_k = int(getattr(args, "stts_top_k", 0) or 0)
            stts_warmup = int(getattr(args, "stts_warmup", 4) or 0)
            if stts_top_k > 0 and turn_id > stts_warmup and turns:
                history = select_stts_history(turns, stts_top_k)
                stts_note = stts_note_text(
                    turn_id=turn_id,
                    total_turns=args.num_turns,
                    all_turns=turns,
                    selected=history,
                )
            prompt = build_claude_prompt(
                item,
                turn_id=turn_id,
                history=history,
                total_turns=args.num_turns,
                stts_note=stts_note,
            )
            turn_kernel_path = output_dir / "kernels" / f"{item.key}_turn_{turn_id}.py"
            turn_raw_path = (
                output_dir / "raw_responses" / f"{item.key}_turn_{turn_id}.md"
            )
            turn_claude_meta: dict[str, Any] = {}
            if args.resume and turn_kernel_path.exists() and turn_raw_path.exists():
                kernel_code = turn_kernel_path.read_text(encoding="utf-8")
                raw_text = turn_raw_path.read_text(encoding="utf-8")
                optimization_notes, optimization_notes_path = (
                    write_turn_optimization_notes(
                        item,
                        output_dir=output_dir,
                        turn_id=turn_id,
                        raw_response=raw_text,
                    )
                )
                turn_claude_meta.update(
                    {
                        "raw_response": raw_text,
                        "optimization_notes": optimization_notes,
                        "optimization_notes_path": optimization_notes_path,
                        "turn_id": turn_id,
                        "raw_response_path": turn_raw_path,
                        "kernel_path": turn_kernel_path,
                    }
                )
            else:
                kernel_code, turn_claude_meta = run_claude(
                    item,
                    output_dir=output_dir,
                    model=args.model,
                    timeout_sec=args.claude_timeout,
                    turn_id=turn_id,
                    prompt=prompt,
                )
                raw_text = str(turn_claude_meta.get("raw_response") or "")
            optimization_notes = str(
                turn_claude_meta.get("optimization_notes")
                or extract_optimization_notes(raw_text)
            )
            claude_meta.update(
                {
                    f"turn_{turn_id}_{key}": value
                    for key, value in turn_claude_meta.items()
                }
            )
            turn_ncu_metrics = None
            turn_enable_nsys = False
            if use_skills:
                turn_ncu_metrics = pending_ncu_metrics or list(FULL_METRIC_SPACE)
            elif use_strategy:
                # nsys every turn (cheap, system view); NCU only in EXPLOIT
                turn_enable_nsys = True
                if phase == "exploit":
                    turn_ncu_metrics = pending_ncu_metrics or list(FULL_METRIC_SPACE)
            with eval_semaphore:
                eval_result = evaluate_kernel(
                    item,
                    kernel_code,
                    server_url=args.server_url,
                    num_correct_trials=args.num_correct_trials,
                    num_perf_trials=args.num_perf_trials,
                    timeout_sec=args.eval_timeout,
                    enable_ncu_profiling=args.enable_ncu_profiling,
                    ncu_metrics=turn_ncu_metrics,
                    enable_nsys_profiling=turn_enable_nsys,
                )
            metrics = drkernel_turn_metrics(eval_result)
            score = score_from_eval(eval_result)
            reward = reward_from_eval(eval_result)
            ncu_report = None
            if use_strategy:
                meta_d = eval_result.get("metadata") or {}
                nsys_payload = meta_d.get("nsys") or {}
                ncu_payload = meta_d.get("ncu") if phase == "exploit" else None
                try:
                    full_report, next_phase, req_metrics, req_packs, analysis_meta = (
                        run_strategy_analysis(
                            item,
                            output_dir=output_dir,
                            model=analysis_model,
                            effort=analysis_effort,
                            timeout_sec=args.claude_timeout,
                            turn_id=turn_id,
                            phase=phase,
                            best_speedup=best_speedup,
                            kernel_code=kernel_code,
                            eval_result=eval_result,
                            nsys_payload=nsys_payload,
                            ncu_payload=ncu_payload,
                        )
                    )
                    ncu_report = re.sub(
                        r"```json\s*.*?```\s*$", "", full_report, flags=re.DOTALL
                    ).strip()[:4000]
                    if next_phase == "exploit":
                        next_list, rejected = resolve_request(req_metrics, req_packs)
                        pending_ncu_metrics = next_list
                    else:
                        pending_ncu_metrics = None
                    print(
                        f"[strategy] {item.key} turn {turn_id}: phase {phase}"
                        f"->{next_phase} score={score:.3f} best={best_speedup:.3f}",
                        flush=True,
                    )
                    phase = next_phase
                    claude_meta.update(
                        {f"turn_{turn_id}_{k}": v for k, v in analysis_meta.items()}
                    )
                except Exception as exc:  # noqa: BLE001 — never block eval
                    print(
                        f"[strategy] {item.key} turn {turn_id} analysis failed: {exc}",
                        flush=True,
                    )
            if score > best_speedup:
                best_speedup = score
            if use_skills:
                ncu_payload = (eval_result.get("metadata") or {}).get("ncu") or {}
                try:
                    full_report, req_metrics, req_packs, analysis_meta = (
                        run_skill_analysis(
                            item,
                            output_dir=output_dir,
                            model=args.model,
                            effort=analysis_effort,
                            timeout_sec=args.claude_timeout,
                            turn_id=turn_id,
                            kernel_code=kernel_code,
                            eval_result=eval_result,
                            ncu_payload=ncu_payload,
                        )
                    )
                    # the trailing ```json collection decision is for the
                    # orchestrator, not the generator — strip it from feedback
                    ncu_report = re.sub(
                        r"```json\s*.*?```\s*$", "", full_report, flags=re.DOTALL
                    ).strip()[:4000]
                    next_list, rejected = resolve_request(req_metrics, req_packs)
                    pending_ncu_metrics = next_list
                    if rejected:
                        print(
                            f"[skill-analysis] {item.key} turn {turn_id}: "
                            f"rejected unknown metrics/packs {rejected}",
                            flush=True,
                        )
                    claude_meta.update(
                        {f"turn_{turn_id}_{k}": v for k, v in analysis_meta.items()}
                    )
                except Exception as exc:  # noqa: BLE001 — analysis must never block eval
                    print(
                        f"[skill-analysis] {item.key} turn {turn_id} failed: {exc}",
                        flush=True,
                    )
                    pending_ncu_metrics = None  # fall back to full space next turn
            turn_row = {
                "turn_id": turn_id,
                "prompt": prompt,
                "raw_response": raw_text,
                "optimization_notes": optimization_notes,
                "kernel_code": kernel_code,
                "score": score,
                "reward": reward,
                "ncu_report": ncu_report,
                "phase": phase if use_strategy else None,
                "metrics": metrics,
                "compiled": eval_result.get("compiled"),
                "correctness": eval_result.get("correctness"),
                "speedup": eval_result.get("speedup"),
                "reference_runtime": eval_result.get("reference_runtime"),
                "kernel_runtime": eval_result.get("kernel_runtime"),
                "decoy_kernel": eval_result.get("decoy_kernel"),
                "error_message": eval_result.get("error_message"),
                "error_code": eval_result.get("error_code"),
                "eval_result": eval_result,
                "state": {
                    "key": item.key,
                    "stage": "evaluated",
                    "turn_id": turn_id,
                    "model": args.model,
                    "server_url": args.server_url,
                    "elapsed_sec": round(time.monotonic() - start, 3),
                    **turn_claude_meta,
                },
                **turn_claude_meta,
            }
            turns.append(turn_row)

        last_turn = turns[-1]
        total_score = round(sum(float(turn.get("score") or 0.0) for turn in turns), 6)
        artifact_paths = write_eval_artifacts(
            item,
            output_dir=output_dir,
            turns=turns,
            total_score=total_score,
        )
        row = {
            "key": item.key,
            "stage": "evaluated",
            "row_index": item.row_index,
            "problem_id": item.problem_id,
            "sample_id": item.sample_id,
            "name": item.name,
            "num_turns": len(turns),
            "prompt": item.prompt_text,
            "reference_code": item.reference_code,
            "raw_response": last_turn.get("raw_response"),
            "optimization_notes": last_turn.get("optimization_notes"),
            "kernel_code": last_turn.get("kernel_code"),
            "score": total_score,
            "last_score": last_turn.get("score"),
            "metrics": last_turn.get("metrics"),
            "turns": turns,
            "elapsed_sec": round(time.monotonic() - start, 3),
            "compiled": last_turn.get("compiled"),
            "correctness": last_turn.get("correctness"),
            "speedup": last_turn.get("speedup"),
            "reference_runtime": last_turn.get("reference_runtime"),
            "kernel_runtime": last_turn.get("kernel_runtime"),
            "decoy_kernel": last_turn.get("decoy_kernel"),
            "error_message": last_turn.get("error_message"),
            "error_code": last_turn.get("error_code"),
            "eval_result": last_turn.get("eval_result"),
            **artifact_paths,
            **claude_meta,
        }
    except Exception as exc:
        row = {
            "key": item.key,
            "stage": "failed",
            "row_index": item.row_index,
            "problem_id": item.problem_id,
            "sample_id": item.sample_id,
            "name": item.name,
            "prompt": item.prompt_text,
            "reference_code": item.reference_code,
            "raw_response": raw_text,
            "num_turns": len(turns),
            "turns": turns,
            "elapsed_sec": round(time.monotonic() - start, 3),
            "error": repr(exc),
            **claude_meta,
        }
    store.write(row)
    return row


def parse_problem_ids(value: str | None) -> set[int] | None:
    if not value:
        return None
    return {int(part.strip()) for part in value.split(",") if part.strip()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Claude Opus 4.8 on KernelBench validation parquet."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--server-url",
        default=os.getenv("KERNELGYM_SERVER_URL")
        or os.getenv("KERNELOGYM_SERVER_URL")
        or DEFAULT_SERVER_URL,
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model", default="claude-opus-4-8")
    parser.add_argument("--sample-id", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--problem-ids", default=None)
    parser.add_argument(
        "--max-claude-workers", type=int, default=1, choices=[1, 2, 3, 4]
    )
    parser.add_argument("--max-eval-workers", type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument("--num-turns", type=int, default=3)
    parser.add_argument(
        "--stts-top-k",
        type=int,
        default=0,
        help="STTS: from turn (warmup+1) on, build the prompt from only the "
        "top-k highest-scoring previous turns (0 = full history)",
    )
    parser.add_argument(
        "--stts-warmup",
        type=int,
        default=4,
        help="STTS: number of initial turns that keep full sequential history",
    )
    parser.add_argument("--claude-timeout", type=int, default=1800)
    parser.add_argument("--eval-timeout", type=int, default=1800)
    parser.add_argument("--num-correct-trials", type=int, default=5)
    parser.add_argument("--num-perf-trials", type=int, default=10)
    parser.add_argument("--enable-ncu-profiling", action="store_true")
    parser.add_argument(
        "--use-skills",
        action="store_true",
        help="hierarchical NCU profiling + skill-driven analysis: collect the "
        "full skill metric space on the first turn, then let the analysis LLM "
        "(gpu-kernel-diag skill) pick next-turn metrics; diagnosis report is "
        "appended to the generation feedback",
    )
    parser.add_argument(
        "--analysis-effort",
        default="high",
        help="effort level for the analysis claude call (default: high)",
    )
    parser.add_argument(
        "--use-strategy",
        action="store_true",
        help="two-stage strategy mode: EXPLORE (nsys + kernel-opt-strategy, "
        "structural/algorithmic search) then EXPLOIT (NCU + gpu-kernel-diag, "
        "local tuning) once the method stabilizes",
    )
    parser.add_argument(
        "--analysis-model",
        default="",
        help="model for the analysis claude call (default: same as --model)",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = default_output_dir(dataset, args.model)
    output_dir.mkdir(parents=True, exist_ok=True)
    grading_dir = (
        output_dir
        if output_dir.name == "grading_results"
        else output_dir / "grading_results"
    )
    grading_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "dataset": dataset,
        "server_url": args.server_url,
        "model": args.model,
        "max_claude_workers": args.max_claude_workers,
        "max_eval_workers": args.max_eval_workers,
        "num_turns": args.num_turns,
        "num_correct_trials": args.num_correct_trials,
        "num_perf_trials": args.num_perf_trials,
        "enable_ncu_profiling": args.enable_ncu_profiling,
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(json_safe(manifest), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (grading_dir / "manifest.json").write_text(
        json.dumps(json_safe(manifest), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    problem_ids = parse_problem_ids(args.problem_ids)
    items = load_items(dataset, args.sample_id, args.limit, problem_ids)
    store = ResultStore(grading_dir)
    eval_semaphore = threading.Semaphore(args.max_eval_workers)

    print(
        f"Running {len(items)} items from {dataset.name} with "
        f"claude_workers={args.max_claude_workers}, eval_workers={args.max_eval_workers}, "
        f"server={args.server_url}",
        flush=True,
    )

    with ThreadPoolExecutor(max_workers=args.max_claude_workers) as executor:
        futures = [
            executor.submit(
                process_one,
                item,
                output_dir=grading_dir,
                store=store,
                args=args,
                eval_semaphore=eval_semaphore,
            )
            for item in items
        ]
        for future in as_completed(futures):
            row = future.result()
            if row.get("stage") == "skipped":
                print(f"{row['key']}: skipped", flush=True)
                continue
            print(
                f"{row['key']}: {row.get('stage')} compiled={row.get('compiled')} "
                f"correct={row.get('correctness')} speedup={row.get('speedup')} "
                f"error={row.get('error') or row.get('error_message')}",
                flush=True,
            )

    summary = json.loads((grading_dir / "summary.json").read_text())
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
