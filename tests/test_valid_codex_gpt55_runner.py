import numpy as np

from valid_codex_gpt55.run_codex_kernelbench import (
    ResultStore,
    WorkItem,
    build_codex_prompt,
    conversation_text,
    drkernel_conversation_row,
    drkernel_turn_metrics,
    extract_python_code,
    prompt_text_from_cell,
)


def test_extract_python_code_prefers_python_fence():
    text = """Some notes.

```python
import torch

class ModelNew:
    pass
```

```text
ignore me
```
"""

    assert extract_python_code(text) == "import torch\n\nclass ModelNew:\n    pass"


def test_extract_python_code_uses_last_modelnew_block_like_drkernel():
    text = """```python
class ModelNew:
    old = True
```

notes

```python
import torch

class ModelNew:
    improved = True
```
"""

    assert (
        extract_python_code(text)
        == "import torch\n\nclass ModelNew:\n    improved = True"
    )


def test_prompt_text_from_numpy_prompt_cell():
    prompt = np.array(
        [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
        ],
        dtype=object,
    )

    assert prompt_text_from_cell(prompt) == "first\n\nsecond"


def test_drkernel_turn_metrics_match_expected_string_shape():
    metrics = drkernel_turn_metrics(
        {
            "compiled": True,
            "correctness": True,
            "speedup": 1.25,
            "decoy_kernel": False,
            "status": "success",
            "metadata": {
                "num_custom_kernels": 2,
                "num_total_kernels": 5,
                "num_coverage": 2,
                "custom_kernel_cuda_time_in_profiling_us": 12.5,
                "total_kernel_run_time_in_profiling_us": 25.0,
            },
        }
    )

    assert metrics["correctness"] == "True"
    assert metrics["compilation"] == "True"
    assert metrics["performance"] == "1.25"
    assert metrics["speedup"] == "1.25"
    assert metrics["is_speedup_positive"] == "True"
    assert metrics["num_custom_kernel"] == "2"
    assert metrics["num_total_kernels"] == "5"
    assert metrics["time_coverage"] == "0.5"


def test_drkernel_conversation_row_uses_one_turn_schema():
    item = WorkItem(
        row_index=0,
        problem_id=7,
        name="example",
        prompt_text="user\nprompt",
        reference_code="class Model: pass",
        sample_id=3,
    )
    row = {
        "key": item.key,
        "raw_response": "```python\nclass ModelNew:\n    pass\n```",
        "score": 1.25,
        "metrics": {"correctness": "True", "speedup": "1.25"},
    }

    conv = drkernel_conversation_row(item, row, global_turn_idx=4)

    assert conv["uid"] == "codex_gpt55_problem_7_sample_3"
    assert conv["num_turns"] == 1
    assert conv["total_score"] == 1.25
    assert conv["turns"][0]["turn_id"] == 1
    assert conv["turns"][0]["global_turn_idx"] == 4
    assert conv["turns"][0]["metrics"]["speedup"] == "1.25"


def test_drkernel_conversation_row_uses_three_turn_schema():
    item = WorkItem(
        row_index=0,
        problem_id=7,
        name="example",
        prompt_text="user\nprompt",
        reference_code="class Model: pass",
        sample_id=3,
    )
    row = {
        "key": item.key,
        "score": 1.5,
        "turns": [
            {
                "turn_id": 1,
                "raw_response": "one",
                "score": 0.5,
                "metrics": {"speedup": "0.5"},
            },
            {
                "turn_id": 2,
                "raw_response": "two",
                "score": 1.0,
                "metrics": {"speedup": "1"},
            },
            {
                "turn_id": 3,
                "raw_response": "three",
                "score": 1.5,
                "metrics": {"speedup": "1.5"},
            },
        ],
    }

    conv = drkernel_conversation_row(item, row, global_turn_idx=10)

    assert conv["num_turns"] == 3
    assert conv["total_score"] == 1.5
    assert [turn["turn_id"] for turn in conv["turns"]] == [1, 2, 3]
    assert [turn["global_turn_idx"] for turn in conv["turns"]] == [10, 11, 12]
    assert conv["turns"][2]["response"] == "three"


def test_build_codex_prompt_later_turn_includes_eval_feedback():
    item = WorkItem(
        row_index=0,
        problem_id=8,
        name="example",
        prompt_text="user\noptimize this",
        reference_code="class Model: pass",
        sample_id=0,
    )

    prompt = build_codex_prompt(
        item,
        turn_id=2,
        previous_turn={
            "kernel_code": "class ModelNew:\n    pass",
            "eval_result": {
                "compiled": True,
                "correctness": False,
                "speedup": 0.0,
                "error_message": "bad",
            },
        },
    )

    assert "Now you have received the server feedback" in prompt
    assert '"correctness": false' in prompt
    assert "bad" in prompt
    assert "class ModelNew" in prompt


def test_result_store_resume_requires_requested_turn_count(tmp_path):
    store = ResultStore(tmp_path)
    store.write(
        {
            "key": "p1_s0",
            "stage": "evaluated",
            "problem_id": 1,
            "sample_id": 0,
            "num_turns": 1,
        }
    )

    assert store.has_completed("p1_s0", requested_turns=1)
    assert not store.has_completed("p1_s0", requested_turns=3)


def test_conversation_text_strips_legacy_role_prefix():
    item = WorkItem(
        row_index=0,
        problem_id=8,
        name="example",
        prompt_text="user\noptimize this",
        reference_code="class Model: pass",
        sample_id=0,
    )

    text = conversation_text(item, [])

    assert text.startswith("[user]\noptimize this")
    assert not text.startswith("[user]\nuser\n")
