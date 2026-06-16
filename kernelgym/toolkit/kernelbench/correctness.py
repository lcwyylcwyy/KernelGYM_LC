"""KernelBench correctness helpers (toolkit layer)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from kernelgym.toolkit.kernelbench.exec_types import (
    KernelExecResult,
    get_error_name,
    set_seed,
)


def register_and_format_exception(
    exception_type: str,
    exception_msg: Exception | str,
    metadata: dict,
    verbose: bool = False,
    truncate: bool = False,
    max_length: int = 200,
):
    if verbose:
        print(f"[Exception {exception_type}] {str(exception_msg)} ")

    metadata[exception_type] = exception_msg
    return metadata


def detect_non_materialized_output(t: Any) -> str:
    """Detect reward-hack outputs that fake their shape without materializing the
    data: a zero-stride broadcast/expand view (e.g. `seed.expand(shape)`) or a
    tensor whose physical storage is far smaller than its logical numel. Such an
    output passes allclose but makes the timed forward near-free (it only writes a
    seed), inflating the measured speedup by 10-100x while doing no real work.
    Returns a reason string if hacked, else "". Tensors inside tuples/lists are
    checked too."""
    if isinstance(t, (tuple, list)):
        for item in t:
            why = detect_non_materialized_output(item)
            if why:
                return why
        return ""
    if not isinstance(t, torch.Tensor):
        return ""
    # zero stride on a dimension of size > 1 => broadcast view, not materialized
    for stride, size in zip(t.stride(), t.shape):
        if size > 1 and stride == 0:
            return (
                f"non-materialized output: zero-stride broadcast view "
                f"(shape {tuple(t.shape)}, strides {tuple(t.stride())})"
            )
    # physical storage holds far fewer elements than the logical tensor =>
    # expand/alias trick that avoids the real write
    try:
        storage_bytes = t.untyped_storage().nbytes()
        storage_elems = storage_bytes // max(t.element_size(), 1)
    except Exception:  # noqa: BLE001
        storage_elems = t.numel()
    if t.numel() > 0 and storage_elems * 4 < t.numel():
        return (
            f"non-materialized output: storage holds {storage_elems} elems "
            f"<< numel {t.numel()} (aliased/expanded, not physically written)"
        )
    return ""


def run_and_check_correctness(
    original_model_instance: nn.Module,
    new_model_instance: nn.Module,
    get_inputs_fn: callable,
    metadata: dict,
    num_correct_trials: int,
    verbose: bool = False,
    seed: int = 42,
    device: Any = None,
) -> KernelExecResult:
    pass_count = 0

    torch.manual_seed(seed)
    correctness_trial_seeds = [
        torch.randint(0, 2**32 - 1, (1,)).item() for _ in range(num_correct_trials)
    ]

    with torch.no_grad():
        for trial in range(num_correct_trials):
            trial_seed = correctness_trial_seeds[trial]
            if verbose:
                print(f"[Eval] Generating Random Input with seed {trial_seed}")

            set_seed(trial_seed)
            inputs = get_inputs_fn()
            inputs = [
                x.cuda(device=device) if isinstance(x, torch.Tensor) else x
                for x in inputs
            ]

            set_seed(trial_seed)
            model = original_model_instance.cuda(device=device)

            set_seed(trial_seed)
            model_new = new_model_instance.cuda(device=device)

            print(f"device: {device}")
            print(f"inputs: {inputs[0].device}")

            output = model(*inputs)
            torch.cuda.synchronize(device=device)

            try:
                output_new = model_new(*inputs)
                torch.cuda.synchronize(device=device)
                if output.shape != output_new.shape:
                    metadata = register_and_format_exception(
                        "correctness_issue",
                        f"Output shape mismatch: Expected {output.shape}, got {output_new.shape}",
                        metadata,
                    )
                    metadata["correctness_issue_name"] = "correctness_issue"
                    if verbose:
                        print(
                            f"[FAIL] trial {trial}: Output shape mismatch: Expected {output.shape}, got {output_new.shape}"
                        )
                    return KernelExecResult(
                        compiled=True, correctness=False, metadata=metadata
                    )

                hack_reason = detect_non_materialized_output(output_new)
                if hack_reason:
                    metadata["correctness_issue"] = "reward_hack"
                    metadata["reward_hack"] = hack_reason
                    metadata["decoy_kernel"] = True
                    if verbose:
                        print(f"[FAIL] trial {trial}: reward hack — {hack_reason}")
                    return KernelExecResult(
                        compiled=True,
                        correctness=False,
                        decoy_kernel=True,
                        metadata=metadata,
                    )

                if not torch.allclose(output, output_new, atol=1e-02, rtol=1e-02):
                    max_diff = torch.max(torch.abs(output - output_new)).item()
                    avg_diff = torch.mean(torch.abs(output - output_new)).item()
                    metadata.setdefault("max_difference", []).append(f"{max_diff:.6f}")
                    metadata.setdefault("avg_difference", []).append(f"{avg_diff:.6f}")
                    metadata["correctness_issue"] = "Output mismatch"
                    if verbose:
                        print(f"[FAIL] trial {trial}: Output mismatch")
                else:
                    pass_count += 1
                    if verbose:
                        print(f"[PASS] trial {trial}: New Model matches Model")

            except Exception as e:
                print("[Error] Exception happens during correctness check")
                print(f"Error in launching kernel for ModelNew: {e}")

                metadata = register_and_format_exception(
                    "runtime_error", e, metadata, truncate=False
                )
                metadata["runtime_error_name"] = get_error_name(e)
                return KernelExecResult(
                    compiled=True, correctness=False, metadata=metadata
                )

    if verbose:
        print(
            f"[Eval] Pass count: {pass_count}, num_correct_trials: {num_correct_trials}"
        )

    metadata["correctness_trials"] = f"({pass_count} / {num_correct_trials})"

    if pass_count == num_correct_trials:
        return KernelExecResult(compiled=True, correctness=True, metadata=metadata)
    return KernelExecResult(compiled=True, correctness=False, metadata=metadata)
