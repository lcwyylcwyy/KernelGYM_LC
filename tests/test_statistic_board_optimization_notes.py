import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATISTIC_BOARD = ROOT / "statistic_board"
sys.path.insert(0, str(STATISTIC_BOARD))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


stat_app = load_module("statistic_board_app", STATISTIC_BOARD / "app.py")
turn_detail = load_module(
    "statistic_board_turn_detail", STATISTIC_BOARD / "turn_detail_dashboard.py"
)


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_eval_outputs_load_visible_optimization_notes(tmp_path):
    grading_results = tmp_path / "grading_results"
    sample_dir = grading_results / "eval_outputs" / "problem_8_sample_0"
    sample_dir.mkdir(parents=True)
    write_json(
        sample_dir / "summary.json",
        {"uid": "codex_gpt55_problem_8_sample_0", "problem_id": 8, "sample_id": 0},
    )
    write_json(
        sample_dir / "turn_1_eval.json",
        {
            "turn_id": 1,
            "status": "completed",
            "compiled": True,
            "correctness": True,
            "speedup": 1.25,
        },
    )
    (sample_dir / "turn_1_kernel.py").write_text(
        "class ModelNew: pass\n", encoding="utf-8"
    )
    (sample_dir / "optimization_notes.txt").write_text(
        "[turn 1]\nOptimization notes:\n- Fuse the epilogue.\n", encoding="utf-8"
    )
    notes_dir = grading_results / "reasoning_summaries"
    notes_dir.mkdir()
    (notes_dir / "p8_s0_turn_1.txt").write_text(
        "Optimization notes:\n- Fuse the epilogue.\n", encoding="utf-8"
    )

    eval_df, detail_map = stat_app.build_eval_outputs_table(
        grading_results / "eval_outputs", {}
    )
    detail_map = stat_app.build_eval_outputs_detail_map(
        grading_results / "eval_outputs",
        detail_map,
        {"codex_gpt55_problem_8_sample_0": {1: "Optimization notes:\n- Fuse.\n"}},
    )

    assert eval_df.iloc[0]["final_speedup"] == 1.25
    assert bool(eval_df.iloc[0]["has_optimization_notes"]) is True
    detail = detail_map["problem_8_sample_0"]
    assert "Fuse the epilogue" in detail["optimization_notes_text"]
    assert detail["turn_items"][0]["optimization_notes"] == (
        "Optimization notes:\n- Fuse the epilogue."
    )


def test_turn_detail_reads_visible_optimization_notes(tmp_path):
    results_dir = tmp_path / "grading_results"
    notes_dir = results_dir / "reasoning_summaries"
    notes_dir.mkdir(parents=True)
    (notes_dir / "p8_s0_turn_2.txt").write_text(
        "Optimization notes:\n- Improve tile size.\n", encoding="utf-8"
    )

    notes = turn_detail.read_visible_optimization_notes(
        results_dir, problem_id=8, sample_idx=0, turn_id=2
    )

    assert notes == "Optimization notes:\n- Improve tile size."


def test_overview_derives_best_by_turn_fast_metrics_from_conversations():
    metrics = {
        "val/test_score/kernelbench_level2_validation_pass@1": 1.0,
        "val/kernel/turn_1/correctness_rate": 0.5,
        "val/kernel/turn_2/correctness_rate": 1.0,
        "val/kernel/turn_3/correctness_rate": 1.0,
    }
    samples = [
        {
            "turns": [
                {"turn_id": 1, "metrics": {"correctness": "True", "speedup": "1.1"}},
                {"turn_id": 2, "metrics": {"correctness": "True", "speedup": "1.3"}},
                {"turn_id": 3, "metrics": {"correctness": "True", "speedup": "1.7"}},
            ]
        },
        {
            "turns": [
                {"turn_id": 1, "metrics": {"correctness": "False", "speedup": "3.0"}},
                {"turn_id": 2, "metrics": {"correctness": "True", "speedup": "0.9"}},
                {"turn_id": 3, "metrics": {"correctness": "True", "speedup": "2.2"}},
            ]
        },
    ]

    overview_df, overview_summary = stat_app.extract_overview_metrics(
        metrics, samples=samples, observed_max_turn=3
    )
    turn_df, best_df = stat_app.extract_fast_trend_data(
        metrics, samples=samples, observed_max_turn=3
    )

    assert "best_by_turn_3" in overview_summary
    fast12_row = overview_df[overview_df["metric"] == "best_by_turn_3 fast@p1.2_in_all"].iloc[0]
    fast15_row = overview_df[overview_df["metric"] == "best_by_turn_3 fast@p1.5_in_all"].iloc[0]
    fast20_row = overview_df[overview_df["metric"] == "best_by_turn_3 fast@p2_in_all"].iloc[0]
    assert fast12_row["value"] == "100.00%"
    assert fast15_row["value"] == "100.00%"
    assert fast20_row["value"] == "50.00%"
    assert set(turn_df["metric"]) == {
        "fast@p1.0_in_all",
        "fast@p1.2_in_all",
        "fast@p1.5_in_all",
        "fast@p2_in_all",
    }
    assert set(best_df["turn"]) == {1, 2, 3}


def test_fast_filter_detail_loads_visible_optimization_notes(tmp_path):
    grading_results = tmp_path / "grading_results"
    sample_dir = grading_results / "eval_outputs" / "problem_8_sample_0"
    sample_dir.mkdir(parents=True)
    (sample_dir / "reference.py").write_text("class Model: pass\n", encoding="utf-8")
    (sample_dir / "optimization_notes.txt").write_text(
        "[turn 2]\nOptimization notes:\n- Use a smaller block.\n", encoding="utf-8"
    )
    notes_dir = grading_results / "reasoning_summaries"
    notes_dir.mkdir()
    (notes_dir / "p8_s0_turn_2.txt").write_text(
        "Optimization notes:\n- Use a smaller block.\n", encoding="utf-8"
    )
    filter_result = {
        "samples": [
            {
                "uid": "codex_gpt55_problem_8_sample_0",
                "problem_id": 8,
                "sample_id": 0,
                "qualified_turn_id": 2,
                "qualified_turn_performance": 1.3,
                "source_eval_output_dir": str(sample_dir),
                "selected_kernel": "class ModelNew: pass",
                "selected_response": "Optimization notes:\n- Use a smaller block.",
            }
        ]
    }

    filter_df, detail_map, _ = stat_app.build_fast_filter_table(filter_result, {})
    detail = detail_map["codex_gpt55_problem_8_sample_0"]

    assert len(filter_df) == 1
    assert "Use a smaller block" in detail["optimization_notes_text"]
    assert detail["selected_optimization_notes"] == (
        "Optimization notes:\n- Use a smaller block."
    )
