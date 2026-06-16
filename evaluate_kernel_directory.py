import argparse
import asyncio
import csv
import importlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

evaluate_pair = importlib.import_module("evaluate_kernel_pair").evaluate_pair


def describe_reference_mode(reference_backend: str) -> str:
    if reference_backend in {"compile", "torch_compile"}:
        return "compile"
    if reference_backend == "eager":
        return "eager"
    return "custom"


def build_reference_label(reference_backend: str) -> str:
    return (
        f"{reference_backend} "
        f"({describe_reference_mode(reference_backend)})"
    )


def render_progress_bar(completed: int, total: int, width: int = 24) -> str:
    if total <= 0:
        return "[" + ("-" * width) + "]"
    filled = int(width * completed / total)
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def print_progress(
    completed: int,
    total: int,
    sample_name: str,
    reference_backend: str,
    reference_mode: str,
) -> None:
    bar = render_progress_bar(completed, total)
    reference_label = build_reference_label(reference_backend)
    max_width = max(shutil.get_terminal_size((120, 20)).columns, 60)
    line = (
        f"{bar} {completed}/{total} "
        f"backend={reference_label} "
        f"sample={sample_name}"
    )
    if len(line) > max_width:
        line = line[: max_width - 3] + "..."

    padded_line = "\r" + line.ljust(max_width)
    stream = sys.stderr
    stream.write(padded_line)
    stream.flush()
    if completed == total:
        stream.write("\n")
        stream.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-validate all selected kernel samples in an exported kernel "
            "directory and report per-sample speedups."
        )
    )
    parser.add_argument(
        "samples_dir",
        type=Path,
        help=(
            "Path to a directory like "
            "best_by_turn_3_fast_at_1_2_in_all_kernels "
            "containing problem_* sample subdirectories."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.2,
        help="Speedup threshold used to flag underperforming samples.",
    )
    parser.add_argument(
        "--reference-backend",
        type=str,
        default="torch_compile",
        help=(
            "Reference execution backend. "
            "Use torch_compile for compile mode."
        ),
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="triton",
        help="Kernel backend name sent to KernelGYM.",
    )
    parser.add_argument(
        "--entry-point",
        type=str,
        default=None,
        help="Optional reference entry point override.",
    )
    parser.add_argument(
        "--num-correct-trials",
        type=int,
        default=5,
        help="Number of correctness trials per sample.",
    )
    parser.add_argument(
        "--num-perf-trials",
        type=int,
        default=100,
        help="Number of performance trials per sample.",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help=(
            "Force the server to re-evaluate instead of using cached results."
        ),
    )
    parser.add_argument(
        "--task-id-prefix",
        type=str,
        default=None,
        help=(
            "Optional task ID prefix for this batch run. When --force-refresh "
            "is set and no prefix is provided, a unique prefix is generated "
            "automatically."
        ),
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="Maximum number of concurrent evaluations.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write the batch report as JSON.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional path to write the per-sample results as CSV.",
    )
    return parser.parse_args()


def load_selected_kernel_path(sample_dir: Path) -> Path:
    summary_path = sample_dir / "source_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing source_summary.json: {summary_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    export_metadata = summary.get("export_metadata") or {}
    selected_kernel_path = export_metadata.get("selected_turn_kernel_path")
    if not selected_kernel_path:
        raise ValueError(
            f"Missing selected_turn_kernel_path in {summary_path}"
        )

    return Path(selected_kernel_path).expanduser().resolve()


def load_source_summary(sample_dir: Path) -> dict[str, Any]:
    summary_path = sample_dir / "source_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing source_summary.json: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def build_fallback_result(
    sample_dir: Path,
    error: Exception,
    reference_backend: str,
) -> dict[str, Any]:
    summary = load_source_summary(sample_dir)
    export_metadata = summary.get("export_metadata") or {}
    speedup = export_metadata.get("selected_turn_speedup")
    reference_mode = describe_reference_mode(reference_backend)
    reference_label = build_reference_label(reference_backend)
    return {
        "sample_dir": str(sample_dir.resolve()),
        "sample_name": sample_dir.name,
        "reference_path": str((sample_dir / "reference.py").resolve()),
        "kernel_path": export_metadata.get("selected_turn_kernel_path"),
        "reference_backend": reference_backend,
        "reference_mode": reference_mode,
        "reference_label": reference_label,
        "compiled": None,
        "correctness": None,
        "speedup": speedup,
        "below_threshold": speedup is None,
        "reference_runtime": None,
        "kernel_runtime": None,
        "error": f"{error} [fallback=export_metadata]",
    }


def build_eval_args(
    base_args: argparse.Namespace,
    sample_dir: Path,
) -> argparse.Namespace:
    reference_path = (sample_dir / "reference.py").resolve()
    kernel_path = load_selected_kernel_path(sample_dir)
    if base_args.task_id_prefix:
        task_id = f"{base_args.task_id_prefix}-{sample_dir.name}"
    else:
        task_id = f"batch-{sample_dir.name}"

    return argparse.Namespace(
        reference_path=reference_path,
        kernel_path=kernel_path,
        reference_backend=base_args.reference_backend,
        backend=base_args.backend,
        entry_point=base_args.entry_point,
        task_id=task_id,
        num_correct_trials=base_args.num_correct_trials,
        num_perf_trials=base_args.num_perf_trials,
        force_refresh=base_args.force_refresh,
    )


async def evaluate_sample(
    base_args: argparse.Namespace,
    sample_dir: Path,
) -> dict[str, Any]:
    eval_args = build_eval_args(base_args, sample_dir)
    result = await evaluate_pair(eval_args)
    speedup = result.get("speedup")
    reference_mode = describe_reference_mode(base_args.reference_backend)
    reference_label = build_reference_label(base_args.reference_backend)
    return {
        "sample_dir": str(sample_dir.resolve()),
        "sample_name": sample_dir.name,
        "reference_path": str(eval_args.reference_path),
        "kernel_path": str(eval_args.kernel_path),
        "reference_backend": base_args.reference_backend,
        "reference_mode": reference_mode,
        "reference_label": reference_label,
        "compiled": result.get("compiled"),
        "correctness": result.get("correctness"),
        "speedup": speedup,
        "below_threshold": speedup is None or speedup < base_args.threshold,
        "reference_runtime": result.get("reference_runtime"),
        "kernel_runtime": result.get("kernel_runtime"),
        "error": result.get("error") or result.get("error_message"),
    }


async def evaluate_all_samples(
    base_args: argparse.Namespace,
    sample_dirs: list[Path],
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(max(1, base_args.max_concurrency))
    total = len(sample_dirs)
    completed = 0
    results: list[dict[str, Any]] = []
    reference_mode = describe_reference_mode(base_args.reference_backend)

    async def run_one(sample_dir: Path) -> dict[str, Any]:
        async with semaphore:
            try:
                return await evaluate_sample(base_args, sample_dir)
            except Exception as exc:
                fallback = build_fallback_result(
                    sample_dir,
                    exc,
                    base_args.reference_backend,
                )
                fallback["below_threshold"] = (
                    fallback["speedup"] is None
                    or fallback["speedup"] < base_args.threshold
                )
                return fallback

    tasks = [run_one(sample_dir) for sample_dir in sample_dirs]
    for task in asyncio.as_completed(tasks):
        result = await task
        results.append(result)
        completed += 1
        print_progress(
            completed,
            total,
            result["sample_name"],
            base_args.reference_backend,
            reference_mode,
        )

    return sorted(results, key=lambda item: item["sample_name"])


def find_sample_dirs(samples_dir: Path) -> list[Path]:
    resolved_dir = samples_dir.expanduser().resolve()
    if not resolved_dir.exists():
        raise FileNotFoundError(f"Samples directory not found: {resolved_dir}")

    return sorted(
        entry
        for entry in resolved_dir.iterdir()
        if entry.is_dir() and entry.name.startswith("problem_")
    )


def print_report(
    results: list[dict[str, Any]],
    threshold: float,
    reference_backend: str,
    task_id_prefix: str | None,
) -> None:
    print(f"Validated {len(results)} samples")
    print(f"reference_label={build_reference_label(reference_backend)}")
    if task_id_prefix:
        print(f"task_id_prefix={task_id_prefix}")
    print(
        "sample_name\treference_label\treference_mode\tspeedup\t"
        "compiled\tcorrectness\tbelow_threshold\terror"
    )
    for item in results:
        speedup = item.get("speedup")
        speedup_text = "N/A" if speedup is None else f"{speedup:.4f}"
        print(
            f"{item['sample_name']}\t{item['reference_label']}\t"
            f"{item['reference_mode']}\t"
            f"{speedup_text}\t{item['compiled']}\t{item['correctness']}\t"
            f"{item['below_threshold']}\t{item['error'] or ''}"
        )

    below = [item for item in results if item["below_threshold"]]
    print()
    print(f"Samples below {threshold:.2f}: {len(below)}")
    for item in below:
        speedup = item.get("speedup")
        speedup_text = "N/A" if speedup is None else f"{speedup:.4f}"
        print(f"- {item['sample_name']}: {speedup_text}")


def write_csv(output_csv: Path, results: list[dict[str, Any]]) -> None:
    fieldnames = [
        "sample_name",
        "sample_dir",
        "reference_path",
        "kernel_path",
        "reference_backend",
        "reference_mode",
        "reference_label",
        "compiled",
        "correctness",
        "speedup",
        "below_threshold",
        "reference_runtime",
        "kernel_runtime",
        "error",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def build_summary(
    results: list[dict[str, Any]],
    threshold: float,
    reference_backend: str,
    task_id_prefix: str | None,
) -> dict[str, Any]:
    below_threshold = [item for item in results if item["below_threshold"]]
    valid_speedups = [
        item["speedup"] for item in results if item["speedup"] is not None
    ]
    return {
        "threshold": threshold,
        "reference_backend": reference_backend,
        "reference_mode": describe_reference_mode(reference_backend),
        "reference_label": build_reference_label(reference_backend),
        "task_id_prefix": task_id_prefix,
        "total_samples": len(results),
        "below_threshold_count": len(below_threshold),
        "below_threshold_samples": below_threshold,
        "min_speedup": min(valid_speedups) if valid_speedups else None,
        "max_speedup": max(valid_speedups) if valid_speedups else None,
        "results": results,
    }


def main() -> int:
    args = parse_args()
    if args.force_refresh and not getattr(args, "task_id_prefix", None):
        args.task_id_prefix = f"batch-{uuid4().hex[:8]}"
    elif not getattr(args, "task_id_prefix", None):
        args.task_id_prefix = None
    sample_dirs = find_sample_dirs(args.samples_dir)
    results = asyncio.run(evaluate_all_samples(args, sample_dirs))
    print_report(
        results,
        args.threshold,
        args.reference_backend,
        args.task_id_prefix,
    )

    if args.output_csv is not None:
        output_csv = args.output_csv.expanduser().resolve()
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        write_csv(output_csv, results)

    if args.output_json is not None:
        output_json = args.output_json.expanduser().resolve()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        summary = build_summary(
            results,
            args.threshold,
            args.reference_backend,
            args.task_id_prefix,
        )
        output_json.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
