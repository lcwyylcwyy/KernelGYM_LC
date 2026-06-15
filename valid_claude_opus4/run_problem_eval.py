#!/usr/bin/env python3
"""Single-process, problem-major KernelBench eval that writes DIRECTLY into the
final output folder (no half dirs, no merge step).

Behaviour:
  - iterate problems in dataset order; for each problem run all NUM_SAMPLES
    samples before moving to the next problem
  - NUM_TURNS turns per sample (default 3)
  - resumable: already-evaluated (problem, sample) cells are skipped
  - output lands in  <OUTPUT_DIR>/grading_results/  in the gpt-5.5 layout
    (eval_outputs/problem_{id}_sample_{s}/..., metrics.json, graded_results.*,
     raw_responses.jsonl, results.jsonl)

Single process => one writer => safe to write the final dir directly.

Defaults reproduce the claude-opus-4-8 matmul-precision run and target:
  drkernel/kernel/scripts/eval/claude-opus-4-8-matmul-precision-2026-06-04/

Everything is env-overridable (KG_*). Examples:
  python valid_claude_opus4/run_problem_eval.py
  KG_NUM_SAMPLES=8 KG_SAMPLE_WORKERS=2 python valid_claude_opus4/run_problem_eval.py
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "valid_claude_opus4"))

from run_claude_kernelbench import (  # noqa: E402
    ResultStore,
    WorkItem,
    load_items,
    process_one,
)


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


DEFAULT_OUTPUT = (
    ROOT / "drkernel/kernel/scripts/eval/claude-opus-4-8-matmul-precision-2026-06-04"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            _env(
                "KG_DATASET",
                str(
                    ROOT
                    / "data/drkernel-validation-data/"
                    "validation_data_thinking_matmul_precision.parquet"
                ),
            )
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(_env("KG_OUTPUT_DIR", str(DEFAULT_OUTPUT))),
    )
    p.add_argument("--server-url", default=_env("KG_SERVER_URL", "http://172.17.0.5:8001"))
    p.add_argument("--model", default=_env("KG_MODEL", "claude-opus-4-8"))
    p.add_argument("--num-samples", type=int, default=int(_env("KG_NUM_SAMPLES", "8")))
    p.add_argument("--num-turns", type=int, default=int(_env("KG_NUM_TURNS", "3")))
    p.add_argument(
        "--sample-workers",
        type=int,
        default=int(_env("KG_SAMPLE_WORKERS", "1")),
        help="parallel samples per problem (1 = sequential)",
    )
    p.add_argument(
        "--eval-workers",
        type=int,
        default=int(_env("KG_EVAL_WORKERS", "1")),
        help="max concurrent server evaluations",
    )
    p.add_argument(
        "--num-correct-trials", type=int, default=int(_env("KG_NUM_CORRECT_TRIALS", "5"))
    )
    p.add_argument(
        "--num-perf-trials", type=int, default=int(_env("KG_NUM_PERF_TRIALS", "10"))
    )
    p.add_argument(
        "--stts-top-k",
        type=int,
        default=int(_env("KG_STTS_TOPK", "0")),
        help="STTS: from turn (warmup+1) on, prompt = top-k scoring previous "
        "turns only (0 = full history)",
    )
    p.add_argument(
        "--stts-warmup",
        type=int,
        default=int(_env("KG_STTS_WARMUP", "4")),
        help="STTS: initial turns that keep full sequential history",
    )
    p.add_argument("--claude-timeout", type=int, default=int(_env("KG_CLAUDE_TIMEOUT", "1800")))
    p.add_argument("--eval-timeout", type=int, default=int(_env("KG_EVAL_TIMEOUT", "1800")))
    p.add_argument("--problem-ids", default=_env("KG_PROBLEM_IDS", "") or None)
    p.add_argument("--enable-ncu-profiling", action="store_true")
    p.add_argument(
        "--use-skills",
        action="store_true",
        default=_env("KG_USE_SKILLS", "") == "1",
        help="hierarchical NCU profiling + skill-driven analysis feedback",
    )
    p.add_argument(
        "--analysis-effort",
        default=_env("KG_ANALYSIS_EFFORT", "high"),
        help="effort level for the analysis claude call",
    )
    p.add_argument(
        "--use-strategy",
        action="store_true",
        default=_env("KG_USE_STRATEGY", "") == "1",
        help="two-stage strategy: EXPLORE (nsys) then EXPLOIT (ncu)",
    )
    p.add_argument(
        "--analysis-model",
        default=_env("KG_ANALYSIS_MODEL", ""),
        help="model for the analysis claude call (default: same as --model)",
    )
    p.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="re-run everything instead of resuming",
    )
    p.set_defaults(resume=True, enable_ncu_profiling=False)
    return p.parse_args()


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    grading_dir = args.output_dir / "grading_results"
    grading_dir.mkdir(parents=True, exist_ok=True)

    problem_ids = None
    if args.problem_ids:
        problem_ids = {int(x) for x in str(args.problem_ids).split(",") if x.strip()}

    # one item per problem (sample_id=0); cloned per sample below
    base_items = load_items(dataset, sample_id=0, limit=None, problem_ids=problem_ids)
    store = ResultStore(grading_dir)
    eval_semaphore = threading.Semaphore(args.eval_workers)

    total_cells = len(base_items) * args.num_samples
    log(f"Output: {grading_dir}")
    log(
        f"dataset={dataset.name} model={args.model} problems={len(base_items)} "
        f"samples={args.num_samples} turns={args.num_turns} "
        f"sample_workers={args.sample_workers} eval_workers={args.eval_workers} "
        f"resume={args.resume} total_cells={total_cells} server={args.server_url}"
    )

    done = 0
    for base in base_items:
        # build the 8 (or N) sample work items for this problem
        sample_items = [
            dataclasses.replace(base, sample_id=s) for s in range(args.num_samples)
        ]

        def _run(item: WorkItem) -> dict:
            return process_one(
                item,
                output_dir=grading_dir,
                store=store,
                args=args,
                eval_semaphore=eval_semaphore,
            )

        if args.sample_workers <= 1:
            for item in sample_items:
                row = _run(item)
                done += 1
                log(
                    f"[{done}/{total_cells}] {row.get('key')}: {row.get('stage')} "
                    f"correct={row.get('correctness')} speedup={row.get('speedup')} "
                    f"err={row.get('error') or row.get('error_message')}"
                )
        else:
            with ThreadPoolExecutor(max_workers=args.sample_workers) as ex:
                futs = {ex.submit(_run, it): it for it in sample_items}
                for fut in as_completed(futs):
                    row = fut.result()
                    done += 1
                    log(
                        f"[{done}/{total_cells}] {row.get('key')}: {row.get('stage')} "
                        f"correct={row.get('correctness')} speedup={row.get('speedup')} "
                        f"err={row.get('error') or row.get('error_message')}"
                    )
        log(f"problem {base.problem_id}: all {args.num_samples} samples done")

    log("ALL DONE")
    summary = (grading_dir / "summary.json")
    if summary.exists():
        log("summary: " + summary.read_text().strip())


if __name__ == "__main__":
    main()
