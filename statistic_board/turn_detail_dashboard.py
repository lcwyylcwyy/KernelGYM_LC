from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd


DEFAULT_RESULTS_DIR = (
    Path("/home/chen/NVS/KernelGYM_LC")
    / "drkernel/kernel/scripts/eval/gpt-5.5-weelinking_0526/grading_results"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_text_file(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace").strip()


def read_visible_optimization_notes(
    results_dir: Path,
    problem_id: Any,
    sample_idx: Any,
    turn_id: Any,
) -> str:
    normalized_problem_id = safe_int(problem_id)
    normalized_sample_idx = safe_int(sample_idx)
    normalized_turn_id = safe_int(turn_id)
    if (
        normalized_problem_id is None
        or normalized_sample_idx is None
        or normalized_turn_id is None
        or normalized_turn_id < 1
    ):
        return ""

    notes_path = (
        results_dir
        / "reasoning_summaries"
        / f"p{normalized_problem_id}_s{normalized_sample_idx}_turn_{normalized_turn_id}.txt"
    )
    return read_text_file(notes_path)


def resolve_results_dir(path_text: str) -> Path:
    path = Path(path_text).expanduser().resolve()
    if path.name != "grading_results" and (path / "grading_results").exists():
        path = path / "grading_results"
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    if not (path / "graded_results_conversations_conversations.jsonl").exists():
        raise FileNotFoundError(f"Missing conversations file under: {path}")
    return path


def safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int | None = None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def extract_code(response: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)```", response or "", flags=re.S)
    if match:
        return match.group(1).strip()
    return ""


def stringify_prompt(prompt: Any) -> str:
    if isinstance(prompt, list):
        parts = []
        for item in prompt:
            if isinstance(item, dict):
                role = item.get("role", "unknown")
                content = item.get("content", "")
                parts.append(f"[{role}]\n{content}")
            else:
                parts.append(str(item))
        return "\n\n".join(parts)
    if isinstance(prompt, dict):
        return json.dumps(prompt, indent=2, ensure_ascii=False)
    return str(prompt or "")


def format_metrics(metrics: dict[str, Any] | None) -> str:
    if not metrics:
        return "{}"
    preferred = [
        "compiled",
        "correctness",
        "success",
        "speedup",
        "performance",
        "status",
        "error",
        "num_custom_kernel",
        "num_total_kernels",
        "time_coverage",
        "ncu_sm_inst_executed_pipe_fma_sum",
        "ncu_sm_inst_executed_sum",
        "ncu_sm_cycles_active_avg",
        "ncu_sm_cycles_elapsed_avg",
        "ncu_l1tex_data_bank_conflicts_shared_ld_sum",
        "ncu_l1tex_t_sector_hit_rate_pct",
        "ncu_warp_stall_barrier_pct",
        "ncu_warp_stall_short_scoreboard_pct",
        "ncu_fma_instruction_ratio",
        "ncu_active_elapsed_cycle_ratio",
    ]
    ordered: dict[str, Any] = {}
    for key in preferred:
        if key in metrics:
            ordered[key] = metrics[key]
    for key in sorted(metrics):
        if key not in ordered:
            ordered[key] = metrics[key]
    return json.dumps(ordered, indent=2, ensure_ascii=False)


def prompt_before_current_response(raw_record: dict[str, Any]) -> str:
    raw_input = raw_record.get("input", "") or ""
    assistant_marker = "\nassistant\n"
    marker_index = raw_input.rfind(assistant_marker)
    if marker_index >= 0:
        return raw_input[:marker_index].rstrip()
    return raw_input


def last_user_message(text: str) -> str:
    user_marker = "\nuser\n"
    marker_index = text.rfind(user_marker)
    if marker_index >= 0:
        text = text[marker_index + len(user_marker) :]
    elif text.startswith("user\n"):
        text = text[len("user\n") :]

    assistant_marker = "\nassistant\n"
    assistant_index = text.find(assistant_marker)
    if assistant_index >= 0:
        text = text[:assistant_index]
    return text.strip()


def server_feedback_after_turn(
    data: "TurnDashboardData",
    problem_id: int,
    sample_idx: int,
    turn_id: int,
    response: str,
) -> str:
    next_record = data.turn_lookup.get((problem_id, sample_idx, turn_id + 1))
    if not next_record:
        return ""

    next_prompt = prompt_before_current_response(next_record.get("raw", {}))
    if response and response in next_prompt:
        tail = next_prompt[next_prompt.rfind(response) + len(response) :]
        feedback = last_user_message(tail)
        if feedback:
            return feedback
    return last_user_message(next_prompt)


def normalized_metric_value(value: Any) -> Any:
    if isinstance(value, str):
        lower_value = value.lower()
        if lower_value == "true":
            return True
        if lower_value == "false":
            return False
        if lower_value == "none":
            return None
        number_value = safe_float(value)
        if number_value is not None:
            return number_value
    return value


def feedback_prompt_from_metrics(metrics: dict[str, Any] | None) -> str:
    if not metrics:
        return ""

    normalized_metrics = {key: normalized_metric_value(value) for key, value in metrics.items()}
    feedback = {
        "status": normalized_metrics.get("status"),
        "compiled": normalized_metrics.get("compiled"),
        "correctness": normalized_metrics.get("correctness"),
        "decoy_kernel": normalized_metrics.get("decoy_kernel"),
        "speedup": normalized_metrics.get("speedup") or normalized_metrics.get("performance"),
        "metadata": normalized_metrics,
        "error_message": normalized_metrics.get("error"),
        "success": normalized_metrics.get("success"),
        "score": normalized_metrics.get("score"),
        "num_custom_kernel": normalized_metrics.get("num_custom_kernel"),
        "num_total_kernels": normalized_metrics.get("num_total_kernels"),
    }
    feedback_json = json.dumps(feedback, indent=2, ensure_ascii=False)
    return (
        "Now you have received the server feedback for your last implementation. "
        "Based on that and all your previous responses, improve the implementation.\n\n"
        "Here is the server feedback. Please refer to this feedback to improve the implementation:\n"
        "Server feedback (status/metrics/errors):\n"
        f"{feedback_json}\n\n"
        "Return an improved Triton implementation named `ModelNew` as a single ```python``` block. "
        "Let's think step by step."
    )


class TurnDashboardData:
    def __init__(self, results_dir: Path, samples_per_problem: int = 8):
        self.results_dir = results_dir
        self.samples_per_problem = samples_per_problem
        self.refs = read_jsonl(results_dir / "graded_results.parquet")
        self.conversations = read_jsonl(results_dir / "graded_results_conversations_conversations.jsonl")
        self.raw_responses = read_jsonl(results_dir / "raw_responses.jsonl")
        self.problem_options: list[str] = []
        self.sample_options_by_problem: dict[int, list[str]] = {}
        self.turn_options_by_sample: dict[tuple[int, int], list[str]] = {}
        self.turn_lookup: dict[tuple[int, int, int], dict[str, Any]] = {}
        self.ref_by_problem: dict[int, dict[str, Any]] = {}
        self.overview_df = pd.DataFrame()
        self.turns_df = pd.DataFrame()
        self._build()

    @staticmethod
    def _reference_snippets(reference_code: str) -> list[str]:
        snippets: list[str] = []
        for raw_line in reference_code.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(("import ", "from ", "#")):
                continue
            if len(line) < 24:
                continue
            snippets.append(line)
            if len(snippets) >= 8:
                break
        return snippets

    def _raw_indices_for_reference(self, reference_code: str) -> list[int]:
        snippets = self._reference_snippets(reference_code)
        if not snippets:
            return []

        min_hits = min(3, len(snippets))
        matches: list[int] = []
        for index, row in enumerate(self.raw_responses):
            prompt = row.get("input", "")
            hits = sum(1 for snippet in snippets if snippet in prompt)
            if hits >= min_hits:
                matches.append(index)
        return matches

    def _build(self) -> None:
        overview_rows: list[dict[str, Any]] = []
        turn_rows: list[dict[str, Any]] = []
        raw_cursor = 0

        for problem_index, ref in enumerate(self.refs):
            info = ref.get("extra_info", {}) or {}
            problem_id = safe_int(info.get("problem_id"), problem_index)
            if problem_id is None:
                problem_id = problem_index
            name = str(info.get("name") or f"problem_{problem_id}")
            self.ref_by_problem[problem_id] = ref
            self.problem_options.append(f"{problem_id}: {name}")
            reference_code = ref.get("reward_model", {}).get("ground_truth", "")
            raw_matches = self._raw_indices_for_reference(reference_code)

            block = self.conversations[
                problem_index * self.samples_per_problem : (problem_index + 1) * self.samples_per_problem
            ]
            sample_options: list[str] = []
            best_speedup = None
            best_sample = None
            best_turn = None
            correct_turns = 0
            total_turns = 0

            for sample_idx, conversation in enumerate(block):
                uid = conversation.get("uid", "")
                total_score = safe_float(conversation.get("total_score"))
                sample_options.append(f"{sample_idx}: {uid}")
                turn_options: list[str] = ["0: initial prompt"]

                for turn in conversation.get("turns", []):
                    turn_id = safe_int(turn.get("turn_id"), len(turn_options) + 1)
                    if turn_id is None:
                        continue
                    metrics = turn.get("metrics") or {}
                    speedup = safe_float(metrics.get("speedup"))
                    correctness = str(metrics.get("correctness", "")).lower() == "true"
                    compiled = str(metrics.get("compiled", "")).lower() == "true"
                    status = metrics.get("status", "")
                    global_turn_idx = safe_int(turn.get("global_turn_idx"))
                    score = safe_float(turn.get("score"))
                    total_turns += 1
                    correct_turns += int(correctness)
                    if speedup is not None and (best_speedup is None or speedup > best_speedup):
                        best_speedup = speedup
                        best_sample = sample_idx
                        best_turn = turn_id

                    turn_options.append(f"{turn_id}: speedup={speedup if speedup is not None else 'NA'}")
                    key = (problem_id, sample_idx, turn_id)
                    raw_match_offset = sample_idx * len(conversation.get("turns", [])) + (turn_id - 1)
                    raw_index = (
                        raw_matches[raw_match_offset]
                        if 0 <= raw_match_offset < len(raw_matches)
                        else raw_cursor
                    )
                    raw_record = self.raw_responses[raw_index] if 0 <= raw_index < len(self.raw_responses) else {}
                    raw_cursor += 1
                    self.turn_lookup[key] = {
                        "problem_index": problem_index,
                        "problem_id": problem_id,
                        "problem_name": name,
                        "sample_idx": sample_idx,
                        "uid": uid,
                        "total_score": total_score,
                        "turn": turn,
                        "metrics": metrics,
                        "global_turn_idx": global_turn_idx,
                        "raw": raw_record,
                        "raw_response_index": raw_index,
                    }
                    row = {
                        "problem_id": problem_id,
                        "problem_name": name,
                        "sample_idx": sample_idx,
                        "turn_id": turn_id,
                        "uid": uid,
                        "global_turn_idx": global_turn_idx,
                        "score": score,
                        "compiled": compiled,
                        "correctness": correctness,
                        "speedup": speedup,
                        "status": status,
                        "error": metrics.get("error"),
                    }
                    for metric_key, metric_value in metrics.items():
                        if str(metric_key).startswith("ncu_"):
                            row[metric_key] = metric_value
                    turn_rows.append(row)

                self.turn_options_by_sample[(problem_id, sample_idx)] = turn_options

            self.sample_options_by_problem[problem_id] = sample_options
            overview_rows.append(
                {
                    "problem_id": problem_id,
                    "name": name,
                    "samples": len(block),
                    "turns": total_turns,
                    "correct_turns": correct_turns,
                    "best_speedup": best_speedup,
                    "best_sample": best_sample,
                    "best_turn": best_turn,
                    "solve_rate": ref.get("solve_rate"),
                }
            )

        self.overview_df = pd.DataFrame(overview_rows).sort_values("problem_id")
        self.turns_df = pd.DataFrame(turn_rows).sort_values(["problem_id", "sample_idx", "turn_id"])


def parse_problem_option(option: str) -> int:
    return int(str(option).split(":", 1)[0])


def parse_index_option(option: str) -> int:
    return int(str(option).split(":", 1)[0])


def load_run(path_text: str, samples_per_problem: int) -> tuple[Any, ...]:
    data = TurnDashboardData(resolve_results_dir(path_text), samples_per_problem=samples_per_problem)
    first_problem = data.problem_options[0] if data.problem_options else None
    first_problem_id = parse_problem_option(first_problem) if first_problem else None
    sample_options = data.sample_options_by_problem.get(first_problem_id, []) if first_problem_id is not None else []
    first_sample = sample_options[0] if sample_options else None
    sample_idx = parse_index_option(first_sample) if first_sample else None
    turn_options = (
        data.turn_options_by_sample.get((first_problem_id, sample_idx), [])
        if first_problem_id is not None and sample_idx is not None
        else []
    )
    first_turn = turn_options[0] if turn_options else None
    status = (
        f"Loaded `{data.results_dir}`\n\n"
        f"- Problems: {len(data.refs)}\n"
        f"- Conversations: {len(data.conversations)}\n"
        f"- Raw prompt records: {len(data.raw_responses)}"
    )
    details = get_turn_detail(data, first_problem, first_sample, first_turn)
    return (
        data,
        status,
        data.overview_df,
        data.turns_df,
        gr.update(choices=data.problem_options, value=first_problem),
        gr.update(choices=sample_options, value=first_sample),
        gr.update(choices=turn_options, value=first_turn),
        *details,
    )


def update_samples(data: TurnDashboardData, problem_option: str) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any]:
    problem_id = parse_problem_option(problem_option)
    sample_options = data.sample_options_by_problem.get(problem_id, [])
    first_sample = sample_options[0] if sample_options else None
    sample_idx = parse_index_option(first_sample) if first_sample else None
    turn_options = data.turn_options_by_sample.get((problem_id, sample_idx), []) if sample_idx is not None else []
    first_turn = turn_options[0] if turn_options else None
    details = get_turn_detail(data, problem_option, first_sample, first_turn)
    return gr.update(choices=sample_options, value=first_sample), gr.update(choices=turn_options, value=first_turn), *details


def update_turns(data: TurnDashboardData, problem_option: str, sample_option: str) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
    problem_id = parse_problem_option(problem_option)
    sample_idx = parse_index_option(sample_option)
    turn_options = data.turn_options_by_sample.get((problem_id, sample_idx), [])
    first_turn = turn_options[0] if turn_options else None
    details = get_turn_detail(data, problem_option, sample_option, first_turn)
    return gr.update(choices=turn_options, value=first_turn), *details


def get_turn_detail(
    data: TurnDashboardData | None,
    problem_option: str | None,
    sample_option: str | None,
    turn_option: str | None,
) -> tuple[str, str, str, str, str, str, str]:
    if data is None or not problem_option or not sample_option or not turn_option:
        return "", "", "", "", "", "", ""

    problem_id = parse_problem_option(problem_option)
    sample_idx = parse_index_option(sample_option)
    user_turn_id = parse_index_option(turn_option)
    ref = data.ref_by_problem.get(problem_id, {})
    reference_code = ref.get("reward_model", {}).get("ground_truth", "")

    assistant_turn_ids = sorted(
        turn_key[2]
        for turn_key in data.turn_lookup
        if turn_key[0] == problem_id and turn_key[1] == sample_idx and turn_key[2] > 0
    )
    if not assistant_turn_ids:
        return "No assistant turns found.", "", reference_code, "", "", "", ""

    first_record = data.turn_lookup[(problem_id, sample_idx, assistant_turn_ids[0])]
    if user_turn_id == 0:
        user_message = prompt_before_current_response(first_record.get("raw", {})) or stringify_prompt(ref.get("prompt"))
    else:
        previous_record = data.turn_lookup.get((problem_id, sample_idx, user_turn_id))
        if previous_record:
            previous_response = previous_record["turn"].get("response", "")
            user_message = server_feedback_after_turn(
                data,
                problem_id,
                sample_idx,
                user_turn_id,
                previous_response,
            )
            if not user_message:
                user_message = feedback_prompt_from_metrics(previous_record.get("metrics"))
        else:
            user_message = ""

    response_turn_id = user_turn_id + 1
    response_record = data.turn_lookup.get((problem_id, sample_idx, response_turn_id))
    if response_record:
        turn = response_record["turn"]
        response = turn.get("response", "")
        metrics_text = format_metrics(response_record.get("metrics"))
        feedback_text = server_feedback_after_turn(
            data,
            problem_id,
            sample_idx,
            response_turn_id,
            response,
        )
        if not feedback_text:
            feedback_text = feedback_prompt_from_metrics(response_record.get("metrics"))
        optimization_notes_text = read_visible_optimization_notes(
            data.results_dir,
            problem_id,
            sample_idx,
            response_turn_id,
        )
    else:
        turn = {}
        response = ""
        metrics_text = "{}"
        feedback_text = ""
        optimization_notes_text = ""

    summary = {
        "problem_id": problem_id,
        "problem_name": first_record["problem_name"],
        "problem_index": first_record["problem_index"],
        "sample_idx": sample_idx,
        "uid": first_record["uid"],
        "user_turn_id": user_turn_id,
        "assistant_response_turn_id": response_turn_id if response_record else None,
        "global_turn_idx": response_record.get("global_turn_idx") if response_record else None,
        "turn_score": turn.get("score"),
        "conversation_total_score": first_record.get("total_score"),
        "solve_rate": ref.get("solve_rate"),
    }
    summary_text = json.dumps(summary, indent=2, ensure_ascii=False)
    return summary_text, metrics_text, reference_code, user_message, response, optimization_notes_text, feedback_text


def build_app(default_results_dir: Path, samples_per_problem: int) -> gr.Blocks:
    with gr.Blocks(title="KernelGYM Turn Detail Dashboard") as app:
        gr.Markdown("# KernelGYM Turn Detail Dashboard")
        data_state = gr.State(None)

        with gr.Row():
            path_box = gr.Textbox(
                label="Run or grading_results path",
                value=str(default_results_dir),
                scale=5,
            )
            samples_box = gr.Number(label="Samples/problem", value=samples_per_problem, precision=0, scale=1)
            load_button = gr.Button("Load", variant="primary")

        status = gr.Markdown()

        with gr.Tab("Overview"):
            overview_table = gr.Dataframe(label="Problems", interactive=False, type="pandas")
            turns_table = gr.Dataframe(label="All Turns", interactive=False, type="pandas")

        with gr.Tab("Turn Detail"):
            with gr.Row():
                problem_dropdown = gr.Dropdown(label="Problem", choices=[])
                sample_dropdown = gr.Dropdown(label="Sample", choices=[])
                turn_dropdown = gr.Dropdown(label="Turn", choices=[])

            with gr.Row():
                summary_box = gr.Textbox(label="Selection Summary", lines=14)
                metrics_box = gr.Textbox(label="Turn Metrics", lines=14)

            with gr.Row():
                reference_box = gr.Textbox(label="Reference Code", lines=24, scale=1)
                prompt_box = gr.Textbox(label="User Message For This Turn", lines=24, scale=1)
                response_box = gr.Textbox(label="Assistant Response To This User Message", lines=24, scale=1)
                optimization_notes_box = gr.Textbox(label="Visible Optimization Notes", lines=24, scale=1)
                feedback_box = gr.Textbox(label="Next Server Feedback", lines=24, scale=1)

        load_outputs = [
            data_state,
            status,
            overview_table,
            turns_table,
            problem_dropdown,
            sample_dropdown,
            turn_dropdown,
            summary_box,
            metrics_box,
            reference_box,
            prompt_box,
            response_box,
            optimization_notes_box,
            feedback_box,
        ]
        load_button.click(
            load_run,
            inputs=[path_box, samples_box],
            outputs=load_outputs,
        )

        problem_dropdown.change(
            update_samples,
            inputs=[data_state, problem_dropdown],
            outputs=[
                sample_dropdown,
                turn_dropdown,
                summary_box,
                metrics_box,
                reference_box,
                prompt_box,
                response_box,
                optimization_notes_box,
                feedback_box,
            ],
        )
        sample_dropdown.change(
            update_turns,
            inputs=[data_state, problem_dropdown, sample_dropdown],
            outputs=[
                turn_dropdown,
                summary_box,
                metrics_box,
                reference_box,
                prompt_box,
                response_box,
                optimization_notes_box,
                feedback_box,
            ],
        )
        turn_dropdown.change(
            get_turn_detail,
            inputs=[data_state, problem_dropdown, sample_dropdown, turn_dropdown],
            outputs=[
                summary_box,
                metrics_box,
                reference_box,
                prompt_box,
                response_box,
                optimization_notes_box,
                feedback_box,
            ],
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--samples-per-problem", type=int, default=8)
    parser.add_argument("--host", default=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("GRADIO_SERVER_PORT", "7862")))
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    app = build_app(args.results_dir, args.samples_per_problem)
    app.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
