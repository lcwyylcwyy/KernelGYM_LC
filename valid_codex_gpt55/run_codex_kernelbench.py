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

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = (
    ROOT
    / "data/drkernel-validation-data/validation_data_thinking_matmul_precision_mini10.parquet"
)
DEFAULT_SERVER_URL = "http://172.17.2.155:8002"
DEFAULT_EVAL_ROOT = ROOT / "drkernel/kernel/scripts/eval"


CODE_FENCE_RE = re.compile(r"```(?:\w+)?\s*\n?(?P<code>.*?)```", re.S)
ROLE_PREFIX_RE = re.compile(r"^(?:user|assistant|system)\n", re.I)


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
        "Codex response did not contain a Python block with class ModelNew"
    )


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
        "uid": f"codex_gpt55_problem_{item.problem_id}_sample_{item.sample_id}",
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
    return DEFAULT_EVAL_ROOT / f"codex-{model}-{suffix}"


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
            uid = f"codex_gpt55_problem_{item.problem_id}_sample_{item.sample_id}"
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
        "val/test_score/kernelbench_level2_validation_pass@1": _rate(speedup_ge_1),
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


def feedback_prompt(eval_result: dict[str, Any]) -> str:
    return f"""Now you have received the server feedback for your last implementation. Based on that and all your previous responses, improve the implementation.

Here is the server feedback. Please refer to this feedback to improve the implementation:
Server feedback (status/metrics/errors):
{json.dumps(json_safe(eval_result), ensure_ascii=False, indent=2)}

Return an improved Triton implementation named `ModelNew` as a single ```python``` block. Let's think step by step.
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
            parts.extend(["", "[user]", feedback_prompt(turn["eval_result"]).rstrip()])
    return "\n".join(parts).rstrip() + "\n"


def build_codex_prompt(
    item: WorkItem,
    *,
    turn_id: int = 1,
    previous_turn: dict[str, Any] | None = None,
    history: list[dict[str, Any]] | None = None,
    total_turns: int = 1,
) -> str:
    if turn_id <= 1:
        return item.prompt_text.rstrip() + "\n"

    turns = list(history or [])
    if not turns and previous_turn:
        turns = [previous_turn]
    if not turns:
        raise ValueError("later turns require previous turn feedback")
    return conversation_text(item, turns)


def run_codex(
    item: WorkItem,
    *,
    output_dir: Path,
    model: str,
    reasoning_effort: str,
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
        "codex",
        "-m",
        model,
        "-c",
        f'model_reasoning_effort="{reasoning_effort}"',
        "--cd",
        str(ROOT),
        "--sandbox",
        "read-only",
        "--ask-for-approval",
        "never",
        "exec",
        "--ephemeral",
        "--ignore-rules",
        "-o",
        str(raw_path),
        "-",
    ]
    start = time.monotonic()
    completed = subprocess.run(
        cmd,
        input=prompt,
        text=True,
        capture_output=True,
        timeout=timeout_sec,
        check=False,
        cwd=ROOT,
    )
    elapsed = time.monotonic() - start
    if completed.returncode != 0:
        raise RuntimeError(
            f"codex exec failed with rc={completed.returncode}: {completed.stderr[-1000:]}"
        )
    raw_text = (
        raw_path.read_text(encoding="utf-8") if raw_path.exists() else completed.stdout
    )
    code = extract_python_code(raw_text)
    kernel_path.write_text(code + "\n", encoding="utf-8")
    meta = {
        "raw_response": raw_text,
        "turn_id": turn_id,
        "codex_returncode": completed.returncode,
        "codex_elapsed_sec": round(elapsed, 3),
        "codex_stdout_tail": completed.stdout[-4000:],
        "codex_stderr_tail": completed.stderr[-4000:],
        "prompt_path": prompt_path,
        "raw_response_path": raw_path,
        "kernel_path": kernel_path,
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
) -> dict[str, Any]:
    task_id = f"codex-gpt55-p{item.problem_id}-s{item.sample_id}-{uuid4().hex[:8]}"
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
        "enable_ncu_profiling": enable_ncu_profiling,
        "run_triton_detection": True,
    }
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
        "summary_path": eval_dir / "summary.json",
    }
    paths["reference_path"].write_text(
        item.reference_code.rstrip() + "\n", encoding="utf-8"
    )
    for turn in turns:
        turn_id = int(turn["turn_id"])
        kernel_path = eval_dir / f"turn_{turn_id}_kernel.py"
        eval_path = eval_dir / f"turn_{turn_id}_eval.json"
        state_path = eval_dir / f"turn_{turn_id}_state.json"
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
        paths[f"turn_{turn_id}_kernel_path"] = kernel_path
        paths[f"turn_{turn_id}_eval_path"] = eval_path
        paths[f"turn_{turn_id}_state_path"] = state_path
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
                    "uid": f"codex_gpt55_problem_{item.problem_id}_sample_{item.sample_id}",
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

    codex_meta: dict[str, Any] = {}
    raw_text = ""
    turns: list[dict[str, Any]] = []
    try:
        for turn_id in range(1, args.num_turns + 1):
            prompt = build_codex_prompt(
                item,
                turn_id=turn_id,
                history=turns,
                total_turns=args.num_turns,
            )
            turn_kernel_path = output_dir / "kernels" / f"{item.key}_turn_{turn_id}.py"
            turn_raw_path = (
                output_dir / "raw_responses" / f"{item.key}_turn_{turn_id}.md"
            )
            turn_codex_meta: dict[str, Any] = {}
            if args.resume and turn_kernel_path.exists() and turn_raw_path.exists():
                kernel_code = turn_kernel_path.read_text(encoding="utf-8")
                raw_text = turn_raw_path.read_text(encoding="utf-8")
            else:
                kernel_code, turn_codex_meta = run_codex(
                    item,
                    output_dir=output_dir,
                    model=args.model,
                    reasoning_effort=args.reasoning_effort,
                    timeout_sec=args.codex_timeout,
                    turn_id=turn_id,
                    prompt=prompt,
                )
                raw_text = str(turn_codex_meta.get("raw_response") or "")
            codex_meta.update(
                {
                    f"turn_{turn_id}_{key}": value
                    for key, value in turn_codex_meta.items()
                }
            )
            with eval_semaphore:
                eval_result = evaluate_kernel(
                    item,
                    kernel_code,
                    server_url=args.server_url,
                    num_correct_trials=args.num_correct_trials,
                    num_perf_trials=args.num_perf_trials,
                    timeout_sec=args.eval_timeout,
                    enable_ncu_profiling=args.enable_ncu_profiling,
                )
            metrics = drkernel_turn_metrics(eval_result)
            score = score_from_eval(eval_result)
            turn_row = {
                "turn_id": turn_id,
                "prompt": prompt,
                "raw_response": raw_text,
                "kernel_code": kernel_code,
                "score": score,
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
                    "reasoning_effort": args.reasoning_effort,
                    "server_url": args.server_url,
                    "elapsed_sec": round(time.monotonic() - start, 3),
                    **turn_codex_meta,
                },
                **turn_codex_meta,
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
            **codex_meta,
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
            **codex_meta,
        }
    store.write(row)
    return row


def parse_problem_ids(value: str | None) -> set[int] | None:
    if not value:
        return None
    return {int(part.strip()) for part in value.split(",") if part.strip()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Codex GPT-5.5 on KernelBench validation parquet."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--server-url",
        default=os.getenv("KERNELGYM_SERVER_URL")
        or os.getenv("KERNELOGYM_SERVER_URL")
        or DEFAULT_SERVER_URL,
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high", "xhigh"],
        default="xhigh",
    )
    parser.add_argument("--sample-id", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--problem-ids", default=None)
    parser.add_argument(
        "--max-codex-workers", type=int, default=2, choices=[1, 2, 3, 4]
    )
    parser.add_argument("--max-eval-workers", type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument("--num-turns", type=int, default=3, choices=[1, 2, 3])
    parser.add_argument("--codex-timeout", type=int, default=1800)
    parser.add_argument("--eval-timeout", type=int, default=1800)
    parser.add_argument("--num-correct-trials", type=int, default=5)
    parser.add_argument("--num-perf-trials", type=int, default=10)
    parser.add_argument("--enable-ncu-profiling", action="store_true")
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
        "reasoning_effort": args.reasoning_effort,
        "max_codex_workers": args.max_codex_workers,
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
        f"codex_workers={args.max_codex_workers}, eval_workers={args.max_eval_workers}, "
        f"server={args.server_url}",
        flush=True,
    )

    with ThreadPoolExecutor(max_workers=args.max_codex_workers) as executor:
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
