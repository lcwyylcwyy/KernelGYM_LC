from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path
from typing import Any


DEFAULT_THRESHOLD = 1.2
METRIC_KEY_PATTERN = re.compile(r"^val/kernel/(best_by_turn|turn)_(\d+)/fast@([^/]+)_in_all$")
METRIC_PREFIX_PATTERN = re.compile(r"^(best_by_turn|turn)(?:_(\d+))?$")
EVAL_OUTPUT_SAMPLE_PATTERN = re.compile(r"sample_(\d+)")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    grading_dir = root / "grading_results"
    parser = argparse.ArgumentParser(
        description=(
            "Export one kernel per problem: the highest-speedup turn that is fast@threshold under the selected metric scope."
        )
    )
    parser.add_argument(
        "--conversations",
        type=Path,
        default=grading_dir / "graded_results_conversations_conversations.jsonl",
        help="Path to graded_results_conversations_conversations.jsonl",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=grading_dir / "metrics.json",
        help="Path to metrics.json",
    )
    parser.add_argument(
        "--eval-outputs",
        type=Path,
        default=grading_dir / "eval_outputs",
        help="Path to eval_outputs directory containing reference kernels and sample summaries",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to store exported kernels. If omitted, generate a metric-based directory name.",
    )
    parser.add_argument(
        "--metric-prefix",
        type=str,
        default="best_by_turn",
        help=(
            "Metric prefix to use: best_by_turn, best_by_turn_<N>, turn, or turn_<N>. "
            "When N is omitted, use --max-turn if provided, otherwise auto-detect from result data."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Speedup threshold used by fast@1.2",
    )
    parser.add_argument(
        "--max-turn",
        type=int,
        default=None,
        help="Max turn included by best_by_turn_N. If omitted, use the maximum turn observed in the result file.",
    )
    return parser.parse_args()


def parse_boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "t"}


def parse_floatish(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    try:
        return float(text)
    except ValueError:
        match = re.search(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", text)
        return float(match.group(0)) if match else default


def extract_code_blocks(response: str) -> list[str]:
    python_blocks = re.findall(r"```python\s*(.*?)```", response, flags=re.IGNORECASE | re.DOTALL)
    if python_blocks:
        return [block.strip() for block in python_blocks if block.strip()]
    generic_blocks = re.findall(r"```(?:[A-Za-z0-9_+-]+)?\s*(.*?)```", response, flags=re.DOTALL)
    return [block.strip() for block in generic_blocks if block.strip()]


def load_metrics_data(metrics_path: Path) -> dict[str, Any]:
    if not metrics_path.exists():
        return {}
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def format_threshold(value: float) -> str:
    return f"{value:g}"


def parse_metric_prefix(metric_prefix: str) -> tuple[str, int | None]:
    text = metric_prefix.strip()
    match = METRIC_PREFIX_PATTERN.fullmatch(text)
    if not match:
        raise ValueError(
            "Invalid --metric-prefix. Use best_by_turn, best_by_turn_<N>, turn, or turn_<N>."
        )
    scope = match.group(1)
    turn_text = match.group(2)
    fixed_turn = int(turn_text) if turn_text else None
    return scope, fixed_turn


def resolve_effective_turn(
    observed_max_turn: int,
    requested_max_turn: int | None,
    fixed_turn_from_prefix: int | None,
) -> int:
    if fixed_turn_from_prefix is not None:
        return fixed_turn_from_prefix
    if requested_max_turn is not None:
        return min(requested_max_turn, observed_max_turn) if observed_max_turn else requested_max_turn
    return observed_max_turn


def build_metric_name(metric_scope: str, target_turn: int, threshold: float) -> str:
    return f"val/kernel/{metric_scope}_{target_turn}/fast@{format_threshold(threshold)}_in_all"


def build_output_stem(metric_scope: str, target_turn: int, threshold: float) -> str:
    threshold_token = format_threshold(threshold).replace(".", "_")
    return sanitize_name(f"{metric_scope}_{target_turn}_fast_at_{threshold_token}_in_all")


def resolve_metric_name_and_value(
    metrics_data: dict[str, Any],
    metric_scope: str,
    target_turn: int,
    threshold: float,
) -> tuple[str, float | None]:
    metric_name = build_metric_name(metric_scope, target_turn, threshold)
    if metric_name in metrics_data:
        return metric_name, parse_floatish(metrics_data[metric_name], default=0.0)

    for key, value in metrics_data.items():
        match = METRIC_KEY_PATTERN.fullmatch(key)
        if not match:
            continue
        metric_scope_in_key = match.group(1)
        if metric_scope_in_key != metric_scope:
            continue
        if int(match.group(2)) != target_turn:
            continue
        metric_threshold = parse_floatish(match.group(3), default=threshold + 1.0)
        if abs(metric_threshold - threshold) < 1e-12:
            return key, parse_floatish(value, default=0.0)

    return metric_name, None


def sanitize_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_sample_dir_name(uid: str, problem_id: Any) -> str:
    if problem_id is None or str(problem_id).strip() == "":
        return sanitize_name(uid)
    return sanitize_name(f"problem_{problem_id}__{uid}")


def load_eval_output_index(eval_outputs_dir: Path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    if not eval_outputs_dir.exists():
        return index

    for entry in sorted(eval_outputs_dir.iterdir()):
        if not entry.is_dir():
            continue

        summary_path = entry / "summary.json"
        if not summary_path.exists():
            continue

        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue

        uid = str(summary.get("uid", "")).strip()
        if not uid:
            continue

        reference_path = entry / "reference.py"
        full_conversation_path = entry / "full_conversation.txt"
        sample_id = summary.get("sample_id")
        if sample_id is None:
            sample_id = summary.get("test_id")
        if sample_id is None:
            match = EVAL_OUTPUT_SAMPLE_PATTERN.search(entry.name)
            sample_id = int(match.group(1)) if match else None

        index[uid] = {
            "problem_id": summary.get("problem_id"),
            "sample_id": sample_id,
            "eval_output_dir": str(entry),
            "eval_output_name": entry.name,
            "summary_source_path": str(summary_path),
            "reference_source_path": str(reference_path) if reference_path.exists() else None,
            "full_conversation_source_path": str(full_conversation_path) if full_conversation_path.exists() else None,
        }

    return index


def write_summary_csv(
    csv_path: Path,
    sample_exports: list[dict[str, Any]],
    metric_name: str,
    metric_scope: str,
    threshold: float,
    target_turn: int,
) -> None:
    fieldnames = [
        "metric_name",
        "metric_scope",
        "threshold",
        "target_turn",
        "uid",
        "problem_id",
        "sample_id",
        "sample_dir",
        "source_eval_output_name",
        "source_eval_output_dir",
        "best_turn_id",
        "best_turn_performance",
        "qualified_turn_id",
        "qualified_turn_performance",
        "generated_kernel_path",
        "generated_response_path",
        "reference_path",
        "full_conversation_path",
        "source_summary_path",
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for sample_export in sample_exports:
            qualified_turns = sample_export.get("qualified_turns") or []
            if not qualified_turns:
                writer.writerow(
                    {
                        "metric_name": metric_name,
                        "metric_scope": metric_scope,
                        "threshold": threshold,
                        "target_turn": target_turn,
                        "uid": sample_export.get("uid"),
                        "problem_id": sample_export.get("problem_id"),
                        "sample_id": sample_export.get("sample_id"),
                        "sample_dir": sample_export.get("sample_dir"),
                        "source_eval_output_name": sample_export.get("source_eval_output_name"),
                        "source_eval_output_dir": sample_export.get("source_eval_output_dir"),
                        "best_turn_id": sample_export.get("best_turn_id"),
                        "best_turn_performance": sample_export.get("best_turn_performance"),
                        "qualified_turn_id": None,
                        "qualified_turn_performance": None,
                        "generated_kernel_path": None,
                        "generated_response_path": None,
                        "reference_path": sample_export.get("reference_path"),
                        "full_conversation_path": sample_export.get("full_conversation_path"),
                        "source_summary_path": sample_export.get("source_summary_path"),
                    }
                )
                continue

            for turn_export in qualified_turns:
                writer.writerow(
                    {
                        "metric_name": metric_name,
                        "metric_scope": metric_scope,
                        "threshold": threshold,
                        "target_turn": target_turn,
                        "uid": sample_export.get("uid"),
                        "problem_id": sample_export.get("problem_id"),
                        "sample_id": sample_export.get("sample_id"),
                        "sample_dir": sample_export.get("sample_dir"),
                        "source_eval_output_name": sample_export.get("source_eval_output_name"),
                        "source_eval_output_dir": sample_export.get("source_eval_output_dir"),
                        "best_turn_id": sample_export.get("best_turn_id"),
                        "best_turn_performance": sample_export.get("best_turn_performance"),
                        "qualified_turn_id": turn_export.get("turn_id"),
                        "qualified_turn_performance": turn_export.get("performance"),
                        "generated_kernel_path": turn_export.get("kernel_path"),
                        "generated_response_path": turn_export.get("response_path"),
                        "reference_path": sample_export.get("reference_path"),
                        "full_conversation_path": sample_export.get("full_conversation_path"),
                        "source_summary_path": sample_export.get("source_summary_path"),
                    }
                )


def export_sample(
    output_dir: Path,
    sample: dict[str, Any],
    qualified_turns: list[dict[str, Any]],
    best_turn_id: int,
    best_performance: float,
    sample_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    uid = str(sample.get("uid", "unknown_uid"))
    metadata = sample_metadata or {}
    sample_dir = output_dir / build_sample_dir_name(uid, metadata.get("problem_id"))
    ensure_dir(sample_dir)

    summary_source_path = metadata.get("summary_source_path")
    reference_source_path = metadata.get("reference_source_path")
    full_conversation_source_path = metadata.get("full_conversation_source_path")
    exported_summary_path = None
    exported_reference_path = None
    exported_full_conversation_path = None

    if summary_source_path:
        summary_src = Path(summary_source_path)
        if summary_src.exists():
            summary_dst = sample_dir / "source_summary.json"
            shutil.copyfile(summary_src, summary_dst)
            exported_summary_path = str(summary_dst)

    if reference_source_path:
        reference_src = Path(reference_source_path)
        if reference_src.exists():
            reference_dst = sample_dir / "reference.py"
            shutil.copyfile(reference_src, reference_dst)
            exported_reference_path = str(reference_dst)

    if full_conversation_source_path:
        full_conversation_src = Path(full_conversation_source_path)
        if full_conversation_src.exists():
            full_conversation_dst = sample_dir / "full_conversation.txt"
            shutil.copyfile(full_conversation_src, full_conversation_dst)
            exported_full_conversation_path = str(full_conversation_dst)

    exported_turns: list[dict[str, Any]] = []
    for turn in qualified_turns:
        turn_id = int(turn["turn_id"])
        performance = float(turn["performance"])
        response = str(turn["response"])
        response_path = sample_dir / f"turn_{turn_id}_response.md"
        response_path.write_text(response, encoding="utf-8")

        code_blocks = extract_code_blocks(response)
        if code_blocks:
            kernel_text = "\n\n".join(code_blocks).strip() + "\n"
            kernel_path = sample_dir / f"turn_{turn_id}_kernel.py"
            kernel_path.write_text(kernel_text, encoding="utf-8")
            kernel_path_str = str(kernel_path)
        else:
            kernel_path_str = None

        exported_turns.append(
            {
                "turn_id": turn_id,
                "performance": performance,
                "response_path": str(response_path),
                "kernel_path": kernel_path_str,
            }
        )

    return {
        "uid": uid,
        "problem_id": metadata.get("problem_id"),
        "sample_id": metadata.get("sample_id", sample.get("sample_id")),
        "sample_dir": str(sample_dir),
        "source_eval_output_dir": metadata.get("eval_output_dir"),
        "source_eval_output_name": metadata.get("eval_output_name"),
        "source_summary_path": exported_summary_path,
        "reference_path": exported_reference_path,
        "full_conversation_path": exported_full_conversation_path,
        "best_turn_id": best_turn_id,
        "best_turn_performance": best_performance,
        "qualified_turns": exported_turns,
    }


def resolve_problem_id(sample: dict[str, Any], sample_metadata: dict[str, Any] | None) -> str:
    metadata = sample_metadata or {}
    problem_id = metadata.get("problem_id")
    if problem_id is None or str(problem_id).strip() == "":
        problem_id = sample.get("problem_id")
    if problem_id is None or str(problem_id).strip() == "":
        problem_id = str(sample.get("uid", "unknown_uid"))
    return str(problem_id)


def iter_qualified_turns_for_sample(
    sample: dict[str, Any],
    metric_scope: str,
    target_turn: int,
    threshold: float,
) -> list[dict[str, Any]]:
    qualified_turns: list[dict[str, Any]] = []
    turns = sample.get("turns", [])
    for turn in turns:
        turn_id = int(turn.get("turn_id", -1))
        if turn_id < 1:
            continue

        if metric_scope == "best_by_turn":
            if turn_id > target_turn:
                continue
        else:
            if turn_id != target_turn:
                continue

        metrics = turn.get("metrics") or {}
        correctness = parse_boolish(metrics.get("correctness"))
        is_decoy = parse_boolish(metrics.get("is_decoy_kernel"))
        performance = parse_floatish(metrics.get("performance"), default=0.0)

        if correctness and not is_decoy and performance >= threshold:
            qualified_turns.append(
                {
                    "turn_id": turn_id,
                    "performance": performance,
                    "response": turn.get("response", ""),
                }
            )

    return qualified_turns


def load_samples_from_conversations(conversations_path: Path) -> tuple[list[dict[str, Any]], int]:
    if not conversations_path.exists():
        raise FileNotFoundError(f"Conversations file not found: {conversations_path}")

    samples: list[dict[str, Any]] = []
    observed_max_turn = 0
    with conversations_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            samples.append(sample)

            for turn in sample.get("turns", []):
                turn_id = int(turn.get("turn_id", -1))
                if turn_id > observed_max_turn:
                    observed_max_turn = turn_id

    return samples, observed_max_turn


def collect_filtered_samples(
    conversations_path: Path,
    metrics_path: Path,
    eval_outputs_path: Path,
    metric_prefix: str = "best_by_turn",
    threshold: float = DEFAULT_THRESHOLD,
    requested_max_turn: int | None = None,
) -> dict[str, Any]:
    metric_scope, fixed_turn_from_prefix = parse_metric_prefix(metric_prefix)
    metrics_data = load_metrics_data(metrics_path)
    eval_output_index = load_eval_output_index(eval_outputs_path)
    samples, observed_max_turn = load_samples_from_conversations(conversations_path)

    target_turn = resolve_effective_turn(
        observed_max_turn=observed_max_turn,
        requested_max_turn=requested_max_turn,
        fixed_turn_from_prefix=fixed_turn_from_prefix,
    )
    if target_turn < 1:
        raise ValueError("Unable to resolve a valid target turn. Check input data or pass --max-turn explicitly.")

    best_candidate_by_problem: dict[str, dict[str, Any]] = {}
    for sample in samples:
        uid = str(sample.get("uid", ""))
        sample_metadata = eval_output_index.get(uid)
        problem_id = resolve_problem_id(sample, sample_metadata)
        qualified_turns = iter_qualified_turns_for_sample(sample, metric_scope, target_turn, threshold)
        if not qualified_turns:
            continue

        best_turn = max(qualified_turns, key=lambda item: float(item["performance"]))
        current = best_candidate_by_problem.get(problem_id)
        if current is None or float(best_turn["performance"]) > float(current["qualified_turn"]["performance"]):
            best_candidate_by_problem[problem_id] = {
                "sample": sample,
                "sample_metadata": sample_metadata,
                "qualified_turn": best_turn,
            }

    selected_samples: list[dict[str, Any]] = []
    for problem_id, candidate in sorted(best_candidate_by_problem.items(), key=lambda item: item[0]):
        sample = candidate["sample"]
        sample_metadata = candidate["sample_metadata"] or {}
        best_turn = candidate["qualified_turn"]
        response = str(best_turn.get("response", ""))
        code_blocks = extract_code_blocks(response)
        selected_kernel = "\n\n".join(code_blocks).strip()

        full_conversation_text = ""
        full_conversation_path = sample_metadata.get("full_conversation_source_path")
        if full_conversation_path and Path(full_conversation_path).exists():
            full_conversation_text = Path(full_conversation_path).read_text(encoding="utf-8", errors="replace")

        selected_samples.append(
            {
                "problem_id": problem_id,
                "uid": str(sample.get("uid", "")),
                "sample_id": sample_metadata.get("sample_id", sample.get("sample_id")),
                "qualified_turn_id": int(best_turn["turn_id"]),
                "qualified_turn_performance": float(best_turn["performance"]),
                "source_eval_output_name": sample_metadata.get("eval_output_name"),
                "source_eval_output_dir": sample_metadata.get("eval_output_dir"),
                "full_conversation_path": full_conversation_path,
                "full_conversation_text": full_conversation_text,
                "selected_kernel": selected_kernel,
                "selected_response": response,
            }
        )

    metric_name = build_metric_name(metric_scope, target_turn, threshold)
    metric_name, metric_value = resolve_metric_name_and_value(metrics_data, metric_scope, target_turn, threshold)

    return {
        "metric_name": metric_name,
        "metric_scope": metric_scope,
        "metric_prefix": metric_prefix,
        "metric_value_in_metrics_json": metric_value,
        "threshold": threshold,
        "target_turn": target_turn,
        "observed_max_turn": observed_max_turn,
        "total_samples": len(samples),
        "selected_problems": len(selected_samples),
        "samples": selected_samples,
    }


def main() -> int:
    args = parse_args()
    script_root = Path(__file__).resolve().parent
    conversations_path: Path = args.conversations
    metrics_path: Path = args.metrics
    eval_outputs_path: Path = args.eval_outputs
    output_dir_arg: Path | None = args.output_dir
    metric_prefix: str = args.metric_prefix
    threshold: float = args.threshold
    requested_max_turn: int | None = args.max_turn

    filter_result = collect_filtered_samples(
        conversations_path=conversations_path,
        metrics_path=metrics_path,
        eval_outputs_path=eval_outputs_path,
        metric_prefix=metric_prefix,
        threshold=threshold,
        requested_max_turn=requested_max_turn,
    )
    metric_scope = str(filter_result["metric_scope"])
    target_turn = int(filter_result["target_turn"])
    observed_max_turn = int(filter_result["observed_max_turn"])

    output_stem = build_output_stem(metric_scope, target_turn, threshold)
    output_dir = output_dir_arg if output_dir_arg is not None else (script_root / f"{output_stem}_kernels")
    ensure_dir(output_dir)

    total_samples = int(filter_result["total_samples"])
    qualified_samples = 0
    qualified_turn_count = 0
    sample_exports: list[dict[str, Any]] = []

    eval_output_index = load_eval_output_index(eval_outputs_path)
    samples_by_uid, _ = load_samples_from_conversations(conversations_path)
    sample_lookup = {str(sample.get("uid", "")): sample for sample in samples_by_uid}

    for selected in filter_result["samples"]:
        uid = str(selected["uid"])
        sample = sample_lookup[uid]
        sample_metadata = eval_output_index.get(uid)
        best_turn = {
            "turn_id": int(selected["qualified_turn_id"]),
            "performance": float(selected["qualified_turn_performance"]),
            "response": str(selected["selected_response"]),
        }
        best_turn_id = int(best_turn["turn_id"])
        best_performance = float(best_turn["performance"])

        qualified_samples += 1
        qualified_turn_count += 1
        sample_exports.append(
            export_sample(
                output_dir=output_dir,
                sample=sample,
                qualified_turns=[best_turn],
                best_turn_id=best_turn_id,
                best_performance=best_performance,
                sample_metadata=sample_metadata,
            )
        )

    metric_name = str(filter_result["metric_name"])
    metric_value = filter_result.get("metric_value_in_metrics_json")

    summary = {
        "metric_name": metric_name,
        "metric_scope": metric_scope,
        "metric_prefix": metric_prefix,
        "metric_value_in_metrics_json": metric_value,
        "threshold": threshold,
        "max_turn": target_turn,
        "target_turn": target_turn,
        "requested_max_turn": requested_max_turn,
        "fixed_turn_from_metric_prefix": parse_metric_prefix(metric_prefix)[1],
        "observed_max_turn": observed_max_turn,
        "effective_max_turn": target_turn,
        "total_samples": total_samples,
        "qualified_samples": qualified_samples,
        "qualified_turns": qualified_turn_count,
        "selection_policy": "best_per_problem",
        "selected_problems": qualified_samples,
        "selected_kernels": qualified_turn_count,
        "qualified_sample_rate": (qualified_samples / total_samples) if total_samples else 0.0,
        "output_dir": str(output_dir),
        "output_stem": output_stem,
        "eval_outputs_dir": str(eval_outputs_path),
        "indexed_eval_outputs": len(eval_output_index),
        "summary_csv_path": str(output_dir.parent / f"{output_stem}_summary.csv"),
        "samples": sample_exports,
    }

    summary_path = output_dir.parent / f"{output_stem}_manifest.json"
    csv_path = output_dir.parent / f"{output_stem}_summary.csv"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary_csv(csv_path, sample_exports, metric_name, metric_scope, threshold, target_turn)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())