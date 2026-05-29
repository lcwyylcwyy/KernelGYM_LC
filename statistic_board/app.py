from __future__ import annotations

import argparse
import json
import os
import platform
import re
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd

import filter as fast_filter

BEST_BY_TURN_PATTERN = re.compile(r"^val/kernel/best_by_turn_(\d+)/")
TURN_EVAL_PATTERN = re.compile(r"^turn_(\d+)_eval\.json$")
TURN_FAST_PATTERN = re.compile(r"^val/kernel/turn_(\d+)/fast@(1(?:\.0|\.2)?)_in_all$")
BEST_BY_TURN_FAST_PATTERN = re.compile(r"^val/kernel/best_by_turn_(\d+)/fast@(1(?:\.0|\.2)?)_in_all$")
DEFAULT_FILTER_THRESHOLD = 1.2
DEFAULT_FILTER_PREFIX = "best_by_turn"


def get_default_run_path() -> str:
    env_path = os.getenv("KERNELGYM_RUN_PATH", "").strip()
    if env_path:
        return env_path

    home = Path.home()
    candidates = [
        # Windows-style common path
        Path(
            r"C:\Users\OT\Downloads\GKG_Eval_Analysis"
            r"\drkernel-8b-maxturns3_9060XT_compile\drkernel-8b-maxturns3_9060XT_compile"
        ),
        # Ubuntu/Linux-style common path
        home
        / "Downloads"
        / "GKG_Eval_Analysis"
        / "drkernel-8b-maxturns3_9060XT_compile"
        / "drkernel-8b-maxturns3_9060XT_compile",
        # If running inside repo root, user can paste/override later
        Path.cwd(),
    ]

    for path in candidates:
        if path.exists():
            return str(path)
    return str(candidates[-1])


DEFAULT_RUN_PATH = get_default_run_path()
DEFAULT_METRIC_PREFIX_CHOICES = ["best_by_turn"]
INDUCTOR_REFERENCE_DIR = Path(__file__).resolve().parent.parent / "data" / "drkernel-validation-data_reference"


def _coerce_nonnegative_int(value: Any) -> int | None:
    try:
        numeric_value = int(value)
    except (TypeError, ValueError):
        return None
    return numeric_value if numeric_value >= 0 else None


def build_inductor_reference_code_index(
    inductor_reference_dir: Path = INDUCTOR_REFERENCE_DIR,
) -> dict[int, dict[str, Any]]:
    code_index: dict[int, dict[str, Any]] = {}
    if not inductor_reference_dir.exists():
        return code_index

    for meta_path in sorted(inductor_reference_dir.glob("*.meta.json")):
        try:
            meta_payload = read_json_file(meta_path)
        except Exception:
            continue

        dashboard_problem_id = _coerce_nonnegative_int(meta_payload.get("row_idx"))
        if dashboard_problem_id is None or dashboard_problem_id in code_index:
            continue

        source_path = meta_path.with_suffix("").with_suffix(".py")
        if not source_path.exists():
            continue

        code_index[dashboard_problem_id] = {
            "row_idx": dashboard_problem_id,
            "meta_problem_id": _coerce_nonnegative_int(meta_payload.get("problem_id")),
            "meta_file": meta_path.name,
            "source_path": str(source_path),
            "code": source_path.read_text(encoding="utf-8", errors="replace"),
        }

    return code_index


def get_inductor_reference_detail(
    problem_id: Any,
    code_index: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    normalized_problem_id = _coerce_nonnegative_int(problem_id)
    if normalized_problem_id is None:
        return {}
    return code_index.get(normalized_problem_id, {})


def get_runtime_info_markdown(initial_run_path: str) -> str:
    system = platform.system() or "unknown"
    release = platform.release() or ""
    pyver = platform.python_version()
    return (
        "### Runtime Info\n"
        f"- OS: **{system} {release}**\n"
        f"- Python: **{pyver}**\n"
        f"- Initial run path: **{initial_run_path}**"
    )


def read_json_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def resolve_grading_results_path(user_input_path: str) -> Path:
    if not user_input_path.strip():
        raise ValueError("Path is empty.")

    root = Path(user_input_path).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {root}")

    if root.name == "grading_results":
        grading_results_dir = root
    elif (root / "grading_results").exists():
        grading_results_dir = root / "grading_results"
    else:
        raise FileNotFoundError(
            "Cannot locate grading_results. Please provide either run root or grading_results path."
        )

    metrics_path = grading_results_dir / "metrics.json"
    eval_outputs_dir = grading_results_dir / "eval_outputs"

    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.json not found: {metrics_path}")
    if not eval_outputs_dir.exists():
        raise FileNotFoundError(f"eval_outputs not found: {eval_outputs_dir}")

    return grading_results_dir


def resolve_filter_inputs(grading_results_dir: Path) -> tuple[Path, Path, Path]:
    conversations_path = grading_results_dir / "graded_results_conversations_conversations.jsonl"
    metrics_path = grading_results_dir / "metrics.json"
    eval_outputs_path = grading_results_dir / "eval_outputs"

    if not conversations_path.exists():
        raise FileNotFoundError(f"Conversations file not found: {conversations_path}")
    return conversations_path, metrics_path, eval_outputs_path


def build_metric_prefix_choices(conversations_path: Path) -> list[str]:
    _, observed_max_turn = fast_filter.load_samples_from_conversations(conversations_path)
    choices = ["best_by_turn"]
    if observed_max_turn > 0:
        choices.extend([f"turn_{turn_idx}" for turn_idx in range(1, observed_max_turn + 1)])
    return choices


def _safe_float(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _format_percent(value: Any) -> str:
    if value is None:
        return "N/A"
    numeric_value = _safe_float(value, fallback=0.0)
    return f"{numeric_value * 100:.2f}%"


def extract_overview_metrics(metrics: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    best_by_turn_numbers: list[int] = []
    for key in metrics:
        matched = BEST_BY_TURN_PATTERN.match(key)
        if matched:
            best_by_turn_numbers.append(int(matched.group(1)))

    latest_turn = max(best_by_turn_numbers) if best_by_turn_numbers else 1

    pass_at_1_key = None
    for key in metrics:
        if "pass@1" in key and "test_score" in key:
            pass_at_1_key = key
            break
    if pass_at_1_key is None:
        for key in metrics:
            if key.endswith("pass@1"):
                pass_at_1_key = key
                break

    fast12_key = f"val/kernel/best_by_turn_{latest_turn}/fast@1.2_in_all"
    fast10_key = f"val/kernel/best_by_turn_{latest_turn}/fast@1_in_all"
    final_correct_key = "val/kernel/final/correctness_rate"

    rows = [
        {
            "metric": "pass@1",
            "value": _format_percent(metrics.get(pass_at_1_key)) if pass_at_1_key else "N/A",
            "source_key": pass_at_1_key or "N/A",
        },
        {
            "metric": f"best_by_turn_{latest_turn} fast@1.2_in_all",
            "value": _format_percent(metrics.get(fast12_key)),
            "source_key": fast12_key,
        },
        {
            "metric": f"best_by_turn_{latest_turn} fast@1.0_in_all",
            "value": _format_percent(metrics.get(fast10_key)),
            "source_key": fast10_key,
        },
        {
            "metric": "final/correctness_rate",
            "value": _format_percent(metrics.get(final_correct_key)),
            "source_key": final_correct_key,
        },
    ]

    df = pd.DataFrame(rows)

    summary_md = (
        "### Overview\n"
        f"- Latest cumulative turn detected: **{latest_turn}**"
    )
    return df, summary_md


def _normalize_fast_threshold_label(label: str) -> str:
    return "1.0" if label == "1" else label


def extract_fast_trend_data(metrics: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    turn_rows: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []

    for key, value in metrics.items():
        turn_match = TURN_FAST_PATTERN.match(key)
        if turn_match:
            turn_rows.append(
                {
                    "turn": int(turn_match.group(1)),
                    "metric": f"fast@{_normalize_fast_threshold_label(turn_match.group(2))}_in_all",
                    "value": _safe_float(value),
                }
            )
            continue

        best_match = BEST_BY_TURN_FAST_PATTERN.match(key)
        if best_match:
            best_rows.append(
                {
                    "turn": int(best_match.group(1)),
                    "metric": f"fast@{_normalize_fast_threshold_label(best_match.group(2))}_in_all",
                    "value": _safe_float(value),
                }
            )

    turn_df = (
        pd.DataFrame(turn_rows, columns=["turn", "metric", "value"])
        if turn_rows
        else pd.DataFrame(columns=["turn", "metric", "value"])
    )
    best_df = (
        pd.DataFrame(best_rows, columns=["turn", "metric", "value"])
        if best_rows
        else pd.DataFrame(columns=["turn", "metric", "value"])
    )

    if not turn_df.empty:
        turn_df = turn_df.sort_values(by=["turn", "metric"], kind="mergesort").reset_index(drop=True)
    if not best_df.empty:
        best_df = best_df.sort_values(by=["turn", "metric"], kind="mergesort").reset_index(drop=True)

    return turn_df, best_df


def _extract_turn_id(path: Path) -> int:
    matched = TURN_EVAL_PATTERN.match(path.name)
    return int(matched.group(1)) if matched else 10**9


def build_eval_outputs_table(
    eval_outputs_dir: Path,
    inductor_reference_code_index: dict[int, dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    detail_map: dict[str, dict[str, Any]] = {}

    for sample_dir in sorted(eval_outputs_dir.iterdir()):
        if not sample_dir.is_dir():
            continue
        if not sample_dir.name.startswith("problem_"):
            continue

        summary_path = sample_dir / "summary.json"
        summary = read_json_file(summary_path) if summary_path.exists() else {}

        turn_eval_files = sorted(sample_dir.glob("turn_*_eval.json"), key=_extract_turn_id)
        turn_evals: list[dict[str, Any]] = []
        for turn_file in turn_eval_files:
            try:
                turn_evals.append(read_json_file(turn_file))
            except Exception:
                continue

        final_turn = turn_evals[-1] if turn_evals else {}
        turn_speedups = [_safe_float(item.get("performance")) for item in turn_evals]
        final_speedup = _safe_float(final_turn.get("performance")) if final_turn else 0.0
        best_speedup = max(turn_speedups) if turn_speedups else 0.0
        speedup_positive_any = any(bool(item.get("is_speedup_positive")) for item in turn_evals)

        problem_id = summary.get("problem_id", final_turn.get("problem_id"))
        sample_id = summary.get("sample_id", final_turn.get("sample_id"))
        eval_status = str(final_turn.get("status", "unknown"))

        conversation_path = sample_dir / "full_conversation.txt"
        if conversation_path.exists():
            conversation_text = conversation_path.read_text(encoding="utf-8", errors="replace")
        else:
            conversation_text = ""

        reference_path = sample_dir / "reference.py"
        if reference_path.exists():
            reference_code = reference_path.read_text(encoding="utf-8", errors="replace")
        else:
            reference_code = ""
        inductor_reference_detail = get_inductor_reference_detail(
            problem_id,
            inductor_reference_code_index,
        )

        row = {
            "problem_id": problem_id,
            "sample_id": sample_id,
            "eval_status": eval_status,
            "num_turns": len(turn_evals),
            "final_speedup": final_speedup,
            "best_speedup": best_speedup,
            "speedup_positive_any": speedup_positive_any,
            "dialogue_log_path": str(conversation_path) if conversation_path.exists() else "",
            "sample_dir_name": sample_dir.name,
        }
        rows.append(row)

        detail_map[sample_dir.name] = {
            "summary": summary,
            "final_turn_eval": final_turn,
            "all_turn_evals": turn_evals,
            "conversation_text": conversation_text,
            "reference_code": reference_code,
            "inductor_reference_code": str(inductor_reference_detail.get("code", "")),
            "inductor_reference_row_idx": inductor_reference_detail.get("row_idx"),
            "inductor_reference_meta_problem_id": inductor_reference_detail.get("meta_problem_id"),
            "inductor_reference_meta_file": str(inductor_reference_detail.get("meta_file", "")),
        }

    if not rows:
        return pd.DataFrame(
            columns=[
                "problem_id",
                "sample_id",
                "eval_status",
                "num_turns",
                "final_speedup",
                "best_speedup",
                "speedup_positive_any",
                "dialogue_log_path",
                "sample_dir_name",
            ]
        ), detail_map

    df = pd.DataFrame(rows)
    df = df.sort_values(by=["problem_id", "sample_id"], kind="mergesort").reset_index(drop=True)
    return df, detail_map


def load_turn_response_index(conversations_path: Path) -> dict[str, dict[int, str]]:
    response_index: dict[str, dict[int, str]] = {}
    samples, _ = fast_filter.load_samples_from_conversations(conversations_path)
    for sample in samples:
        uid = str(sample.get("uid", "")).strip()
        if not uid:
            continue
        turn_response_map: dict[int, str] = {}
        for turn in sample.get("turns", []):
            try:
                turn_id = int(turn.get("turn_id", -1))
            except (TypeError, ValueError):
                continue
            if turn_id < 1:
                continue
            turn_response_map[turn_id] = str(turn.get("response", ""))
        response_index[uid] = turn_response_map
    return response_index


def build_eval_outputs_detail_map(
    eval_outputs_dir: Path,
    base_detail_map: dict[str, dict[str, Any]],
    response_index: dict[str, dict[int, str]],
) -> dict[str, dict[str, Any]]:
    enriched = dict(base_detail_map)

    for sample_dir_name, detail in list(enriched.items()):
        sample_dir = eval_outputs_dir / sample_dir_name
        if not sample_dir.exists():
            continue

        summary = detail.get("summary", {})
        uid = str(summary.get("uid", "")).strip()
        if not uid:
            turn_eval_files = sorted(sample_dir.glob("turn_*_eval.json"), key=_extract_turn_id)
            if turn_eval_files:
                try:
                    first_turn_eval = read_json_file(turn_eval_files[0])
                    uid = str(first_turn_eval.get("uid", "")).strip()
                except Exception:
                    uid = ""

        turn_responses = response_index.get(uid, {})
        turn_items: list[dict[str, Any]] = []
        for turn_eval_file in sorted(sample_dir.glob("turn_*_eval.json"), key=_extract_turn_id):
            try:
                turn_eval = read_json_file(turn_eval_file)
            except Exception:
                continue

            try:
                turn_id = int(turn_eval.get("turn_id", -1))
            except (TypeError, ValueError):
                continue
            if turn_id < 1:
                continue

            kernel_path = sample_dir / f"turn_{turn_id}_kernel.py"
            if kernel_path.exists():
                kernel_code = kernel_path.read_text(encoding="utf-8", errors="replace")
            else:
                kernel_code = ""

            turn_items.append(
                {
                    "turn_id": turn_id,
                    "score": _safe_float(turn_eval.get("score")),
                    "speedup": _safe_float(turn_eval.get("performance")),
                    "correctness": bool(turn_eval.get("correctness", False)),
                    "compiled": bool(turn_eval.get("compilation", False)),
                    "kernel_code": kernel_code,
                    "response": turn_responses.get(turn_id, ""),
                }
            )

        turn_items.sort(key=lambda item: int(item["turn_id"]))
        detail["turn_items"] = turn_items
        detail["uid"] = uid

    return enriched


def build_turn_metrics_summary(turn_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    for item in turn_items:
        summary_rows.append(
            {
                "turn_id": item.get("turn_id"),
                "score": item.get("score"),
                "speedup": item.get("speedup"),
                "correctness": item.get("correctness"),
                "compiled": item.get("compiled"),
            }
        )
    return summary_rows


def _pick_turn_payload(detail: dict[str, Any], turn_id: int | None) -> tuple[str, str, int | None, list[int]]:
    turn_items = detail.get("turn_items", [])
    if not turn_items:
        return "", "", None, []

    all_turn_ids = [int(item.get("turn_id", -1)) for item in turn_items if int(item.get("turn_id", -1)) >= 1]
    if not all_turn_ids:
        return "", "", None, []

    if turn_id is None or turn_id not in all_turn_ids:
        turn_id = all_turn_ids[0]

    selected = next((item for item in turn_items if int(item.get("turn_id", -1)) == turn_id), None)
    if selected is None:
        return "", "", None, all_turn_ids

    return str(selected.get("kernel_code", "")), str(selected.get("response", "")), turn_id, all_turn_ids


def build_turn_payload_map(turn_items: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    payload_map: dict[str, dict[str, str]] = {}
    for item in turn_items:
        try:
            turn_id = int(item.get("turn_id", -1))
        except (TypeError, ValueError):
            continue
        if turn_id < 1:
            continue
        payload_map[str(turn_id)] = {
            "kernel_code": str(item.get("kernel_code", "")),
            "response": str(item.get("response", "")),
        }
    return payload_map


def build_fast_filter_table(
    filter_result: dict[str, Any],
    inductor_reference_code_index: dict[int, dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], str]:
    rows: list[dict[str, Any]] = []
    detail_map: dict[str, dict[str, Any]] = {}

    for item in filter_result.get("samples", []):
        uid = str(item.get("uid", ""))
        problem_id = item.get("problem_id")
        sample_id = item.get("sample_id")
        selected_kernel = str(item.get("selected_kernel", ""))
        full_conversation_text = str(item.get("full_conversation_text", ""))

        rows.append(
            {
                "problem_id": problem_id,
                "sample_id": sample_id,
                "uid": uid,
                "qualified_turn_id": item.get("qualified_turn_id"),
                "speedup": item.get("qualified_turn_performance"),
                "source_eval_output_name": item.get("source_eval_output_name"),
            }
        )

        reference_code = ""
        source_eval_output_dir = item.get("source_eval_output_dir")
        if source_eval_output_dir:
            reference_path = Path(str(source_eval_output_dir)) / "reference.py"
            if reference_path.exists():
                reference_code = reference_path.read_text(encoding="utf-8", errors="replace")
        inductor_reference_detail = get_inductor_reference_detail(
            problem_id,
            inductor_reference_code_index,
        )

        detail_map[uid] = {
            "item": item,
            "conversation_text": full_conversation_text,
            "selected_kernel": selected_kernel,
            "selected_response": str(item.get("selected_response", "")),
            "reference_code": reference_code,
            "inductor_reference_code": str(inductor_reference_detail.get("code", "")),
            "inductor_reference_row_idx": inductor_reference_detail.get("row_idx"),
            "inductor_reference_meta_problem_id": inductor_reference_detail.get("meta_problem_id"),
            "inductor_reference_meta_file": str(inductor_reference_detail.get("meta_file", "")),
        }

    if rows:
        df = pd.DataFrame(rows).sort_values(by=["problem_id", "sample_id"], kind="mergesort").reset_index(drop=True)
    else:
        df = pd.DataFrame(
            columns=[
                "problem_id",
                "sample_id",
                "uid",
                "qualified_turn_id",
                "speedup",
                "source_eval_output_name",
            ]
        )

    summary_md = (
        "### Fast@1.2 Filter Summary\n"
        f"- Metric: **{filter_result.get('metric_name', 'N/A')}**\n"
        f"- Threshold: **{filter_result.get('threshold', DEFAULT_FILTER_THRESHOLD)}**\n"
        f"- Target turn: **{filter_result.get('target_turn', 'N/A')}**\n"
        f"- Selected problems: **{filter_result.get('selected_problems', 0)} / {filter_result.get('total_samples', 0)}**"
    )
    return df, detail_map, summary_md


def _normalize_metric_prefix(metric_prefix: str, choices: list[str]) -> str:
    candidate = (metric_prefix or "").strip()
    if candidate and candidate in choices:
        return candidate
    return choices[0] if choices else "best_by_turn"


def load_dashboard_data(run_path: str, threshold: float, metric_prefix: str):
    input_path_display = run_path.strip() if run_path and run_path.strip() else "<empty>"
    try:
        grading_results = resolve_grading_results_path(run_path)
        metrics = read_json_file(grading_results / "metrics.json")
        overview_df, overview_summary = extract_overview_metrics(metrics)
        turn_fast_df, best_by_turn_fast_df = extract_fast_trend_data(metrics)
        inductor_reference_code_index = build_inductor_reference_code_index()

        eval_outputs_dir = grading_results / "eval_outputs"
        eval_df, detail_map = build_eval_outputs_table(eval_outputs_dir, inductor_reference_code_index)
        conversations_path, metrics_path, eval_outputs_path = resolve_filter_inputs(grading_results)
        response_index = load_turn_response_index(conversations_path)
        detail_map = build_eval_outputs_detail_map(eval_outputs_dir, detail_map, response_index)
        metric_prefix_choices = build_metric_prefix_choices(conversations_path)
        metric_prefix_value = _normalize_metric_prefix(metric_prefix, metric_prefix_choices)
        filter_result = fast_filter.collect_filtered_samples(
            conversations_path=conversations_path,
            metrics_path=metrics_path,
            eval_outputs_path=eval_outputs_path,
            metric_prefix=metric_prefix_value,
            threshold=float(threshold),
        )
        filter_df, filter_detail_map, filter_summary_md = build_fast_filter_table(
            filter_result,
            inductor_reference_code_index,
        )

        status_md = (
            "### Loaded Successfully\n"
            f"- input path: **{input_path_display}**\n"
            f"- grading_results: **{grading_results}**\n"
            f"- total samples: **{len(eval_df)}**"
        )
        return (
            status_md,
            overview_summary,
            overview_df,
            turn_fast_df,
            best_by_turn_fast_df,
            eval_df,
            detail_map,
            filter_summary_md,
            filter_df,
            filter_detail_map,
            gr.update(choices=metric_prefix_choices, value=metric_prefix_value),
        )
    except Exception as exc:
        empty_overview = pd.DataFrame(columns=["metric", "value", "source_key"])
        empty_eval = pd.DataFrame(
            columns=[
                "problem_id",
                "sample_id",
                "eval_status",
                "num_turns",
                "final_speedup",
                "best_speedup",
                "speedup_positive_any",
                "dialogue_log_path",
                "sample_dir_name",
            ]
        )
        empty_filter = pd.DataFrame(
            columns=[
                "problem_id",
                "sample_id",
                "uid",
                "qualified_turn_id",
                "speedup",
                "source_eval_output_name",
            ]
        )
        empty_trend = pd.DataFrame(columns=["turn", "metric", "value"])
        status_md = (
            "### Failed to Load\n"
            f"- input path: **{input_path_display}**\n"
            f"- error: **{exc}**"
        )
        return (
            status_md,
            "",
            empty_overview,
            empty_trend,
            empty_trend,
            empty_eval,
            {},
            "",
            empty_filter,
            {},
            gr.update(choices=DEFAULT_METRIC_PREFIX_CHOICES, value=DEFAULT_METRIC_PREFIX_CHOICES[0]),
        )


def show_sample_detail(evt: gr.SelectData, eval_df: pd.DataFrame, detail_map: dict[str, dict[str, Any]]):
    if eval_df is None or len(eval_df) == 0:
        return {}, "", "", "", gr.update(choices=[], value=None), "", "", {}

    row_index = evt.index[0] if isinstance(evt.index, (list, tuple)) else int(evt.index)
    if row_index < 0 or row_index >= len(eval_df):
        return {}, "", "", "", gr.update(choices=[], value=None), "", "", {}

    sample_dir_name = str(eval_df.iloc[row_index]["sample_dir_name"])
    detail = detail_map.get(sample_dir_name, {})

    turn_items = detail.get("turn_items", [])
    detail_json = {
        "sample_dir_name": sample_dir_name,
        "inductor_lookup_problem_id": _coerce_nonnegative_int(
            eval_df.iloc[row_index]["problem_id"]
        ),
        "inductor_reference_row_idx": _coerce_nonnegative_int(
            detail.get("inductor_reference_row_idx")
        ),
        "inductor_reference_meta_problem_id": _coerce_nonnegative_int(
            detail.get("inductor_reference_meta_problem_id")
        ),
        "inductor_reference_meta_file": detail.get("inductor_reference_meta_file", ""),
        "turn_metrics": build_turn_metrics_summary(turn_items),
    }
    reference_code = detail.get("reference_code", "")
    inductor_reference_code = detail.get("inductor_reference_code", "")
    conversation_text = detail.get("conversation_text", "")
    kernel_code, response_text, selected_turn_id, all_turn_ids = _pick_turn_payload(detail, turn_id=None)
    turn_dropdown_update = gr.update(choices=all_turn_ids, value=selected_turn_id)
    turn_payload_map = build_turn_payload_map(turn_items)
    return (
        detail_json,
        reference_code,
        inductor_reference_code,
        conversation_text,
        turn_dropdown_update,
        kernel_code,
        response_text,
        turn_payload_map,
    )


def show_eval_turn_content(
    turn_id: int | None,
    turn_payload_map: dict[str, dict[str, str]],
) -> tuple[str, str]:
    if not turn_payload_map:
        return "", ""

    if turn_id is None:
        selected_key = sorted(turn_payload_map.keys(), key=lambda item: int(item))[0]
    else:
        selected_key = str(int(turn_id))
        if selected_key not in turn_payload_map:
            selected_key = sorted(turn_payload_map.keys(), key=lambda item: int(item))[0]

    selected = turn_payload_map.get(selected_key, {})
    return str(selected.get("kernel_code", "")), str(selected.get("response", ""))


def show_filtered_detail(evt: gr.SelectData, filter_df: pd.DataFrame, filter_detail_map: dict[str, dict[str, Any]]):
    if filter_df is None or len(filter_df) == 0:
        return {}, "", "", "", "", ""

    row_index = evt.index[0] if isinstance(evt.index, (list, tuple)) else int(evt.index)
    if row_index < 0 or row_index >= len(filter_df):
        return {}, "", "", "", "", ""

    uid = str(filter_df.iloc[row_index]["uid"])
    detail = filter_detail_map.get(uid, {})
    item = detail.get("item", {})
    detail_json = {
        "uid": uid,
        "problem_id": item.get("problem_id"),
        "inductor_lookup_problem_id": _coerce_nonnegative_int(item.get("problem_id")),
        "inductor_reference_row_idx": _coerce_nonnegative_int(
            detail.get("inductor_reference_row_idx")
        ),
        "inductor_reference_meta_problem_id": _coerce_nonnegative_int(
            detail.get("inductor_reference_meta_problem_id")
        ),
        "inductor_reference_meta_file": detail.get("inductor_reference_meta_file", ""),
        "sample_id": item.get("sample_id"),
        "qualified_turn_id": item.get("qualified_turn_id"),
        "qualified_turn_performance": item.get("qualified_turn_performance"),
        "source_eval_output_name": item.get("source_eval_output_name"),
        "full_conversation_path": item.get("full_conversation_path"),
    }
    return (
        detail_json,
        detail.get("reference_code", ""),
        detail.get("inductor_reference_code", ""),
        detail.get("conversation_text", ""),
        detail.get("selected_kernel", ""),
        detail.get("selected_response", ""),
    )


def build_app(initial_run_path: str = DEFAULT_RUN_PATH) -> gr.Blocks:
    with gr.Blocks(title="KernelGYM Statistic Board") as app:
        gr.Markdown("## KernelGYM Statistic Board")
        runtime_info_md = gr.Markdown(value=get_runtime_info_markdown(initial_run_path))

        with gr.Row():
            run_path_box = gr.Textbox(label="Run Path", value=initial_run_path, scale=6)
            threshold_box = gr.Number(label="Fast Threshold", value=DEFAULT_FILTER_THRESHOLD, precision=2, scale=1)
            metric_prefix_box = gr.Dropdown(
                label="Metric Prefix",
                choices=DEFAULT_METRIC_PREFIX_CHOICES,
                value=DEFAULT_METRIC_PREFIX_CHOICES[0],
                interactive=True,
                scale=2,
            )
            load_button = gr.Button("Load", scale=1)

        status_md = gr.Markdown()

        with gr.Tab("Overview"):
            overview_summary_md = gr.Markdown()
            overview_table = gr.Dataframe(label="Key Metrics", interactive=False, type="pandas")
            turn_fast_plot = gr.LinePlot(
                x="turn",
                y="value",
                color="metric",
                title="Per-Turn Fast@1.0/1.2 In All",
                x_title="Turn",
                y_title="Value",
                sort="x",
                x_axis_format="d",
                tooltip=["turn", "metric", "value"],
            )
            best_by_turn_fast_plot = gr.LinePlot(
                x="turn",
                y="value",
                color="metric",
                title="Best-By-Turn Fast@1.0/1.2 In All",
                x_title="Turn",
                y_title="Value",
                sort="x",
                x_axis_format="d",
                tooltip=["turn", "metric", "value"],
            )

        with gr.Tab("Eval Outputs"):
            eval_table = gr.Dataframe(label="All Samples", interactive=False, type="pandas")
            with gr.Row():
                with gr.Column(scale=1):
                    detail_json = gr.JSON(label="Selected Sample Detail")
                with gr.Column(scale=2):
                    reference_code_box = gr.Code(
                        label="Reference Code (reference.py)",
                        language="python",
                        interactive=False,
                    )
                    inductor_reference_code_box = gr.Code(
                        label="Inductor Triton Code",
                        language="python",
                        interactive=False,
                    )
                    conversation_box = gr.Textbox(
                        label="Dialogue Log",
                        lines=24,
                        max_lines=36,
                    )
            with gr.Row():
                eval_turn_selector = gr.Dropdown(
                    label="Turn",
                    choices=[],
                    value=None,
                    interactive=True,
                )
            with gr.Row():
                eval_turn_kernel_box = gr.Code(
                    label="Selected Turn Kernel",
                    language="python",
                    interactive=False,
                )
                eval_turn_response_box = gr.Textbox(
                    label="Selected Turn Response",
                    lines=16,
                    max_lines=24,
                )

        with gr.Tab("Fast@1.2 Filter"):
            filter_summary_md = gr.Markdown()
            filter_table = gr.Dataframe(label="Selected Fast@1.2 Samples", interactive=False, type="pandas")
            with gr.Row():
                filtered_detail_json = gr.JSON(label="Filtered Sample Detail")
            with gr.Row():
                with gr.Column(scale=1):
                    filtered_reference_code_box = gr.Code(
                        label="Reference Code (reference.py)",
                        language="python",
                        interactive=False,
                    )
                    filtered_inductor_reference_code_box = gr.Code(
                        label="Inductor Triton Code",
                        language="python",
                        interactive=False,
                    )
                    filtered_conversation_box = gr.Textbox(
                        label="Full Conversation",
                        lines=14,
                        max_lines=20,
                    )
                    filtered_response_box = gr.Textbox(
                        label="Selected Response",
                        lines=10,
                        max_lines=14,
                    )
                with gr.Column(scale=1):
                    filtered_kernel_box = gr.Code(
                        label="Selected Optimized Kernel",
                        language="python",
                        interactive=False,
                    )

        detail_state = gr.State({})
        filter_detail_state = gr.State({})
        eval_turn_payload_state = gr.State({})

        load_button.click(
            fn=load_dashboard_data,
            inputs=[run_path_box, threshold_box, metric_prefix_box],
            outputs=[
                status_md,
                overview_summary_md,
                overview_table,
                turn_fast_plot,
                best_by_turn_fast_plot,
                eval_table,
                detail_state,
                filter_summary_md,
                filter_table,
                filter_detail_state,
                metric_prefix_box,
            ],
        )

        eval_table.select(
            fn=show_sample_detail,
            inputs=[eval_table, detail_state],
            outputs=[
                detail_json,
                reference_code_box,
                inductor_reference_code_box,
                conversation_box,
                eval_turn_selector,
                eval_turn_kernel_box,
                eval_turn_response_box,
                eval_turn_payload_state,
            ],
        )

        eval_turn_selector.change(
            fn=show_eval_turn_content,
            inputs=[eval_turn_selector, eval_turn_payload_state],
            outputs=[eval_turn_kernel_box, eval_turn_response_box],
        )

        filter_table.select(
            fn=show_filtered_detail,
            inputs=[filter_table, filter_detail_state],
            outputs=[
                filtered_detail_json,
                filtered_reference_code_box,
                filtered_inductor_reference_code_box,
                filtered_conversation_box,
                filtered_kernel_box,
                filtered_response_box,
            ],
        )

        app.load(
            fn=load_dashboard_data,
            inputs=[run_path_box, threshold_box, metric_prefix_box],
            outputs=[
                status_md,
                overview_summary_md,
                overview_table,
                turn_fast_plot,
                best_by_turn_fast_plot,
                eval_table,
                detail_state,
                filter_summary_md,
                filter_table,
                filter_detail_state,
                metric_prefix_box,
            ],
        )

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KernelGYM statistic board")
    parser.add_argument("--run-path", default=DEFAULT_RUN_PATH, help="Run root path or grading_results path")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", default=7860, type=int, help="Port to bind")
    parser.add_argument("--share", action="store_true", help="Enable Gradio share URL")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = build_app(initial_run_path=args.run_path)
    app.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
