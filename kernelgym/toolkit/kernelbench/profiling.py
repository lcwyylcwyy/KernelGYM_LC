"""KernelBench profiling helpers (toolkit layer)."""

from __future__ import annotations

import logging
import os
import csv
import io
import json
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

from kernelgym.config import settings
from kernelgym.config.ncu_metrics import DEFAULT_NCU_METRICS

logger = logging.getLogger("kernelgym.toolkit.kernelbench.profiling")

NCU_SCALAR_ALIASES = {
    "sm__inst_executed_pipe_fma.sum": ("ncu_sm_inst_executed_pipe_fma_sum", "sum"),
    "sm__inst_executed.sum": ("ncu_sm_inst_executed_sum", "sum"),
    "sm__cycles_active.avg": ("ncu_sm_cycles_active_avg", "avg"),
    "sm__cycles_elapsed.avg": ("ncu_sm_cycles_elapsed_avg", "avg"),
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum": (
        "ncu_l1tex_data_bank_conflicts_shared_ld_sum",
        "sum",
    ),
    "l1tex__t_sector_hit_rate.pct": ("ncu_l1tex_t_sector_hit_rate_pct", "avg"),
    "smsp__warp_issue_stalled_barrier_per_warp_active.pct": (
        "ncu_warp_stall_barrier_pct",
        "avg",
    ),
    "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct": (
        "ncu_warp_stall_short_scoreboard_pct",
        "avg",
    ),
}


def normalize_ncu_metrics(metrics: Optional[Union[str, Sequence[str]]]) -> List[str]:
    """Return a clean metric list from config/request values."""
    if metrics is None or metrics == "":
        return list(DEFAULT_NCU_METRICS)
    if isinstance(metrics, str):
        try:
            parsed = json.loads(metrics)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except Exception:
            pass
        return [item.strip() for item in metrics.split(",") if item.strip()]
    return [str(item).strip() for item in metrics if str(item).strip()]


def _parse_ncu_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text or text.lower() in {"n/a", "na", "nan", "--"}:
        return None
    if text.endswith("%"):
        text = text[:-1].strip()
    try:
        return float(text)
    except ValueError:
        return None


def _row_value(row: Dict[str, Any], *names: str) -> Any:
    lowered = {str(key).strip().lower(): value for key, value in row.items()}
    for name in names:
        value = lowered.get(name.lower())
        if value is not None:
            return value
    return None


def _aggregate_ncu_value(metric_name: str, metric_payload: Dict[str, Any]) -> Optional[float]:
    if not metric_payload.get("count"):
        return None
    if metric_name.endswith(".sum"):
        return metric_payload.get("sum")
    if metric_name.endswith(".avg") or metric_name.endswith(".pct"):
        return metric_payload.get("avg")
    return metric_payload.get("avg")


def _append_ncu_metric(
    grouped: Dict[str, Dict[str, Any]],
    metric_name: str,
    value: float,
    unit: Any = "",
    kernel_name: Any = "",
) -> None:
    payload = grouped.setdefault(
        metric_name,
        {
            "unit": str(unit or ""),
            "count": 0,
            "sum": 0.0,
            "avg": 0.0,
            "min": value,
            "max": value,
            "values": [],
            "per_kernel": [],
        },
    )
    payload["count"] += 1
    payload["sum"] += value
    payload["min"] = min(payload["min"], value)
    payload["max"] = max(payload["max"], value)
    payload["values"].append(value)
    payload["per_kernel"].append(
        {
            "kernel_name": str(kernel_name or ""),
            "value": value,
            "unit": payload["unit"],
        }
    )


def _finalize_ncu_metrics(grouped: Dict[str, Dict[str, Any]]) -> None:
    for metric_name, payload in grouped.items():
        count = int(payload.get("count") or 0)
        payload["avg"] = payload["sum"] / count if count else 0.0
        payload["value"] = _aggregate_ncu_value(metric_name, payload)


def build_ncu_scalar_aliases(metrics: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    scalars: Dict[str, float] = {}
    for metric_name, (alias, reducer) in NCU_SCALAR_ALIASES.items():
        metric_payload = metrics.get(metric_name)
        if not metric_payload:
            continue
        value = metric_payload.get(reducer)
        if value is not None:
            scalars[alias] = float(value)

    fma = scalars.get("ncu_sm_inst_executed_pipe_fma_sum")
    inst = scalars.get("ncu_sm_inst_executed_sum")
    if fma is not None and inst and inst > 0:
        scalars["ncu_fma_instruction_ratio"] = float(fma / inst)

    active = scalars.get("ncu_sm_cycles_active_avg")
    elapsed = scalars.get("ncu_sm_cycles_elapsed_avg")
    if active is not None and elapsed and elapsed > 0:
        scalars["ncu_active_elapsed_cycle_ratio"] = float(active / elapsed)

    return scalars


def _parse_ncu_wide_csv(
    lines: List[str],
    header_index: int,
    metrics_filter: set[str],
) -> Dict[str, Dict[str, Any]]:
    reader = csv.reader(io.StringIO("\n".join(lines[header_index:])))
    try:
        header = next(reader)
    except StopIteration:
        return {}

    metric_indices = {
        metric_name: idx
        for idx, metric_name in enumerate(header)
        if metric_name in metrics_filter
    }
    if not metric_indices:
        return {}

    try:
        unit_row = next(reader)
    except StopIteration:
        unit_row = []

    id_index = header.index("ID") if "ID" in header else None
    kernel_index = header.index("Kernel Name") if "Kernel Name" in header else None
    grouped: Dict[str, Dict[str, Any]] = {}

    for row in reader:
        if id_index is not None:
            row_id = row[id_index].strip() if id_index < len(row) else ""
            if not row_id:
                continue
        kernel_name = row[kernel_index] if kernel_index is not None and kernel_index < len(row) else ""
        for metric_name, metric_index in metric_indices.items():
            raw_value = row[metric_index] if metric_index < len(row) else None
            value = _parse_ncu_number(raw_value)
            if value is None:
                continue
            unit = unit_row[metric_index] if metric_index < len(unit_row) else ""
            _append_ncu_metric(grouped, metric_name, value, unit, kernel_name)

    return grouped


def _parse_ncu_long_csv(
    lines: List[str],
    header_index: int,
    metrics_filter: set[str],
) -> Dict[str, Dict[str, Any]]:
    reader = csv.DictReader(io.StringIO("\n".join(lines[header_index:])))
    grouped: Dict[str, Dict[str, Any]] = {}
    for row in reader:
        metric_name = _row_value(row, "Metric Name", "Metric")
        if not metric_name:
            continue
        metric_name = str(metric_name).strip()
        if metrics_filter and metric_name not in metrics_filter:
            continue

        value = _parse_ncu_number(_row_value(row, "Metric Value", "Value"))
        if value is None:
            continue

        unit = _row_value(row, "Metric Unit", "Unit")
        kernel_name = _row_value(row, "Kernel Name", "Kernel")
        _append_ncu_metric(grouped, metric_name, value, unit, kernel_name)

    return grouped


def parse_ncu_csv(
    csv_text: str,
    requested_metrics: Optional[Union[str, Sequence[str]]] = None,
) -> Dict[str, Any]:
    """Parse Nsight Compute raw CSV into aggregate per-metric payloads."""
    metrics_filter = set(normalize_ncu_metrics(requested_metrics))
    lines = [line for line in (csv_text or "").splitlines() if line.strip()]
    header_index = None
    wide_header_index = None
    for idx, line in enumerate(lines):
        if "Metric Name" in line and "Metric Value" in line:
            header_index = idx
            break
        try:
            fields = next(csv.reader([line]))
        except Exception:
            continue
        if "Kernel Name" in fields and metrics_filter.intersection(fields):
            wide_header_index = idx
            break

    if header_index is None and wide_header_index is None:
        return {
            "status": "failed",
            "metrics": {},
            "scalars": {},
            "error": "NCU CSV header not found",
            "requested_metrics": sorted(metrics_filter),
        }

    if wide_header_index is not None:
        grouped = _parse_ncu_wide_csv(lines, wide_header_index, metrics_filter)
    else:
        grouped = _parse_ncu_long_csv(lines, header_index or 0, metrics_filter)

    _finalize_ncu_metrics(grouped)

    scalars = build_ncu_scalar_aliases(grouped)
    missing_metrics = sorted(metrics_filter - set(grouped.keys())) if metrics_filter else []
    return {
        "status": "ok" if grouped else "failed",
        "metrics": grouped,
        "scalars": scalars,
        "requested_metrics": sorted(metrics_filter),
        "missing_metrics": missing_metrics,
    }


def _device_arg(device: Union[torch.device, int, str, None]) -> str:
    if device is None:
        return "cuda:0"
    if isinstance(device, torch.device):
        if device.type == "cuda":
            return f"cuda:{device.index if device.index is not None else 0}"
        return str(device)
    if isinstance(device, int):
        return f"cuda:{device}"
    return str(device)


def _write_ncu_driver_script(script_path: Path) -> None:
    script_path.write_text(
        r'''
from __future__ import annotations

import sys
from pathlib import Path

import torch

from kernelgym.toolkit.kernelbench.exec_types import set_seed
from kernelgym.toolkit.kernelbench.loading import (
    graceful_eval_cleanup,
    load_custom_model_with_tempfile,
    load_original_model_and_inputs,
)


def _to_cuda(values, device):
    return [
        value.cuda(device=device) if isinstance(value, torch.Tensor) else value
        for value in values
    ]


def main():
    reference_path = Path(sys.argv[1])
    kernel_path = Path(sys.argv[2])
    entry_point = sys.argv[3]
    device = torch.device(sys.argv[4])
    seed_num = int(sys.argv[5])
    num_warmup = int(sys.argv[6])
    num_trials = int(sys.argv[7])

    torch.cuda.set_device(device)
    context = {}
    tempfile_handle = None
    try:
        reference_src = reference_path.read_text(encoding="utf-8")
        kernel_src = kernel_path.read_text(encoding="utf-8")
        Model, get_init_inputs, get_inputs = load_original_model_and_inputs(
            reference_src,
            context,
            entry_point,
        )

        set_seed(seed_num)
        init_inputs = _to_cuda(get_init_inputs(), device)
        if (
            len(init_inputs) > 1
            and hasattr(init_inputs[0], "__len__")
            and not isinstance(init_inputs[0], (str, torch.Tensor))
            and len(init_inputs[0]) == 0
        ):
            init_inputs = init_inputs[1]

        ModelNew, tempfile_handle = load_custom_model_with_tempfile(
            kernel_src,
            entry_point=f"{entry_point}New",
        )
        with torch.no_grad():
            set_seed(seed_num)
            model_new = ModelNew(*init_inputs) if isinstance(init_inputs, list) else ModelNew(**init_inputs)
            model_new = model_new.cuda(device=device)
            inputs = _to_cuda(get_inputs(), device)
            torch.cuda.synchronize(device=device)
            for _ in range(num_warmup):
                model_new(*inputs)
                torch.cuda.synchronize(device=device)
            for _ in range(num_trials):
                model_new(*inputs)
            torch.cuda.synchronize(device=device)
    finally:
        graceful_eval_cleanup(context, device, tempfile_handle)


if __name__ == "__main__":
    main()
'''.lstrip(),
        encoding="utf-8",
    )


def _excerpt(text: str, max_chars: int = 4000) -> str:
    text = text or ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars // 2] + "\n... [truncated] ...\n" + text[-max_chars // 2 :]


def run_ncu_profiling(
    *,
    original_model_src: str,
    custom_model_src: str,
    entry_point: str,
    device: Union[torch.device, int, str, None],
    metrics: Optional[Union[str, Sequence[str]]] = None,
    seed_num: int = 42,
    num_warmup: int = 1,
    num_trials: int = 1,
    timeout_sec: int = 300,
) -> Dict[str, Any]:
    """Run Nsight Compute in a child Python process and parse raw CSV output."""
    metric_list = normalize_ncu_metrics(metrics)
    ncu_path = shutil.which("ncu")
    if not ncu_path:
        return {
            "status": "failed",
            "metrics": {},
            "scalars": {},
            "requested_metrics": metric_list,
            "error": "ncu executable not found on PATH",
        }

    with tempfile.TemporaryDirectory(prefix="kgym_ncu_") as temp_dir:
        temp_path = Path(temp_dir)
        reference_path = temp_path / "reference.py"
        kernel_path = temp_path / "kernel.py"
        script_path = temp_path / "run_ncu_target.py"
        reference_path.write_text(original_model_src, encoding="utf-8")
        kernel_path.write_text(custom_model_src, encoding="utf-8")
        _write_ncu_driver_script(script_path)

        cmd = [
            ncu_path,
            "--csv",
            "--page",
            "raw",
            "--metrics",
            ",".join(metric_list),
            sys.executable,
            str(script_path),
            str(reference_path),
            str(kernel_path),
            entry_point,
            _device_arg(device),
            str(seed_num),
            str(num_warmup),
            str(num_trials),
        ]
        env = os.environ.copy()
        repo_root = str(Path(__file__).resolve().parents[3])
        if env.get("PYTHONPATH"):
            env["PYTHONPATH"] = os.pathsep.join([repo_root, env["PYTHONPATH"]])
        else:
            env["PYTHONPATH"] = repo_root
        try:
            completed = subprocess.run(
                cmd,
                cwd=os.getcwd(),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                check=False,
            )
        except Exception as exc:
            return {
                "status": "failed",
                "metrics": {},
                "scalars": {},
                "requested_metrics": metric_list,
                "error": str(exc),
            }

    parsed = parse_ncu_csv(completed.stdout, requested_metrics=metric_list)
    parsed["returncode"] = completed.returncode
    parsed["stderr"] = _excerpt(completed.stderr)
    if completed.returncode != 0:
        parsed["status"] = "failed"
        parsed["error"] = parsed.get("error") or _excerpt(completed.stderr or completed.stdout)
    return parsed


def compute_triton_kernel_coverage(matched_triton_kernels: List[str], profilling_result: Dict[str, Any]):
    """Compute the coverage of the matched triton kernels in the profiling result."""

    def _matches_profiler_name(captured: str, profiler_name: str) -> bool:
        cap = captured.lower()
        prof = profiler_name.lower()
        if cap == prof:
            return True
        if cap in prof or prof in cap:
            return True
        return False

    kernels = matched_triton_kernels
    num_custom_kernels = 0
    kernel_names = [kernel.split(" ")[0] for kernel in kernels]

    kernels_in_profiling = profilling_result["kernels"]

    total_time = 0.0
    matched_cuda_time = 0.0
    triton_kernels_in_profiling = []

    for prof_kernel in kernels_in_profiling:
        prof_name = prof_kernel["name"]
        cuda_time = float(prof_kernel["cuda_time_us"])
        cpu_time = float(prof_kernel["cpu_time_us"])
        total_time += cuda_time + cpu_time

        if any(_matches_profiler_name(kernel_name, prof_name) for kernel_name in kernel_names):
            triton_kernels_in_profiling.append(prof_name)
            num_custom_kernels += 1
            matched_cuda_time += cuda_time

    triton_kernels_not_in_profiling = [
        kernel_name
        for kernel_name in kernel_names
        if not any(_matches_profiler_name(kernel_name, prof_name) for prof_name in triton_kernels_in_profiling)
    ]

    return {
        "num_custom_kernels": num_custom_kernels,
        "num_total_kernels": len(kernels_in_profiling),
        "total_kernel_run_time_in_profiling_us": total_time,
        "custom_kernel_cuda_time_in_profiling_us": matched_cuda_time,
        "triton_kernels_not_in_profiling": triton_kernels_not_in_profiling,
        "triton_kernels_in_profiling": triton_kernels_in_profiling,
    }


@contextmanager
def profiling_context(enabled: bool = True):
    if not enabled:
        yield None
        return

    try:
        import torch.profiler as profiler

        activities = []
        if "cpu" in settings.profiling_activities:
            activities.append(profiler.ProfilerActivity.CPU)
        if "cuda" in settings.profiling_activities:
            activities.append(profiler.ProfilerActivity.CUDA)

        print(f"[Profiler] Initializing with activities: {[str(a) for a in activities]}")

        if not activities:
            print("[Profiler] No activities configured, profiler will return no data")
            yield None
            return

        prof = profiler.profile(
            activities=activities,
            record_shapes=settings.profiling_record_shapes,
            profile_memory=settings.profiling_profile_memory,
            with_stack=settings.profiling_with_stack,
            on_trace_ready=None,
        )

        prof.__enter__()
        try:
            print("[Profiler] Profiler started successfully")
            cuda_available = torch.cuda.is_available()
            cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            device_info = "cuda:unavailable"
            if cuda_available:
                try:
                    current_device = torch.cuda.current_device()
                    device_name = torch.cuda.get_device_name(current_device)
                    device_info = f"cuda:{current_device} ({device_name})"
                except Exception as e:
                    device_info = f"cuda:unknown (error={e})"
            print(
                "[Profiler] Context pid=%s cuda_available=%s device=%s CUDA_VISIBLE_DEVICES=%s",
                os.getpid(),
                cuda_available,
                device_info,
                cuda_visible,
            )
            if cuda_available:
                try:
                    test = torch.ones((1024,), device="cuda")
                    _ = test.sum()
                    torch.cuda.synchronize()
                    print("[Profiler] Self-test CUDA op executed")
                except Exception as e:
                    print(f"[Profiler] Self-test failed: {e}")
            yield prof
        finally:
            try:
                prof.__exit__(None, None, None)
                print("[Profiler] Profiler stopped successfully")
            except Exception as e:
                print(f"[Profiler] Error during profiler cleanup: {e}")

    except Exception as e:
        logger.warning(f"[Profiler] Failed to initialize profiler: {e}. Continuing without profiling.")
        yield None


def extract_profiling_metrics(prof: Optional["torch.profiler.profile"]) -> Dict[str, Any]:
    if prof is None:
        return {}

    try:
        import torch.profiler as profiler

        events = prof.key_averages()
        print(f"[Profiler] key_averages: {events}")
        total_events = len(events)
        cuda_device_event_count = 0
        cuda_time_event_count = 0
        self_cuda_time_event_count = 0

        logger.debug(f"[Profiler] Captured {total_events} total events")

        def _safe_metric(evt: Any, names: Tuple[str, ...], default: float = 0.0) -> float:
            for name in names:
                if hasattr(evt, name):
                    value = getattr(evt, name)
                    if callable(value):
                        try:
                            value = value()
                        except Exception:
                            continue
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        continue
            return default

        def _safe_int_metric(evt: Any, names: Tuple[str, ...], default: int = 0) -> int:
            for name in names:
                if hasattr(evt, name):
                    value = getattr(evt, name)
                    if callable(value):
                        try:
                            value = value()
                        except Exception:
                            continue
                    try:
                        return int(value)
                    except (TypeError, ValueError):
                        continue
            return default

        cuda_kernels = []
        total_cpu_time = 0.0
        total_self_cuda_time = 0.0
        for evt in events:
            cpu_time_us = _safe_metric(evt, ("cpu_time_total", "cpu_time"), 0.0)
            total_cpu_time += cpu_time_us

            cuda_time_us = _safe_metric(
                evt,
                ("device_time_total", "device_time", "cuda_time_total", "cuda_time"),
                0.0,
            )
            self_cuda_time_us = _safe_metric(
                evt,
                ("self_cuda_time_total", "self_cuda_time"),
                0.0,
            )
            if self_cuda_time_us > 0.0:
                self_cuda_time_event_count += 1
                total_self_cuda_time += self_cuda_time_us
            if cuda_time_us <= 0.0:
                continue
            device_type = getattr(evt, "device_type", None)
            if device_type is not None and device_type != profiler.DeviceType.CUDA:
                pass
            elif device_type == profiler.DeviceType.CUDA:
                cuda_device_event_count += 1
            cuda_time_event_count += 1

            kernel_entry = {
                "name": getattr(evt, "key", "unknown"),
                "cuda_time_us": cuda_time_us,
                "cpu_time_us": cpu_time_us,
                "count": _safe_int_metric(evt, ("count",), 0),
            }
            memory_usage = _safe_metric(evt, ("cuda_memory_usage",), 0.0)
            if memory_usage > 0.0:
                kernel_entry["cuda_memory_usage"] = memory_usage
            cuda_kernels.append(kernel_entry)

        cuda_kernels.sort(key=lambda x: x["cuda_time_us"], reverse=True)

        logger.debug(
            f"[Profiler] Filtered to {len(cuda_kernels)} CUDA kernels (from {len(events)} total)"
        )
        if len(cuda_kernels) == 0 and len(events) > 0:
            logger.warning(
                f"[Profiler] Captured events but no CUDA kernels! Event types: {[getattr(evt, 'device_type', 'unknown') for evt in list(events)[:5]]}"
            )

        memory_stats = {}
        try:
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                memory_stats = {
                    "allocated_mb": torch.cuda.memory_allocated(device) / (1024 * 1024),
                    "reserved_mb": torch.cuda.memory_reserved(device) / (1024 * 1024),
                    "max_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024 * 1024),
                    "max_reserved_mb": torch.cuda.max_memory_reserved(device) / (1024 * 1024),
                }
        except Exception as e:
            logger.warning(f"[Profiler] Failed to collect memory stats: {e}")

        profiling_metrics = {
            "kernels": cuda_kernels,
            "kernel_count": len(cuda_kernels),
            "total_cpu_time_us": total_cpu_time,
            "total_cuda_time_us": sum(k["cuda_time_us"] for k in cuda_kernels),
            "total_self_cuda_time_us": total_self_cuda_time,
            "cuda_device_event_count": cuda_device_event_count,
            "cuda_time_event_count": cuda_time_event_count,
            "self_cuda_time_event_count": self_cuda_time_event_count,
            "memory_stats": memory_stats,
        }

        if len(cuda_kernels) == 0:
            profiling_metrics["profiling_warning"] = (
                "Profiler captured no CUDA kernels. This may indicate a profiler failure."
            )

        return profiling_metrics

    except Exception as e:
        logger.warning(f"[Profiler] Failed to extract profiling metrics: {e}")
        return {"profiling_error": str(e)}
