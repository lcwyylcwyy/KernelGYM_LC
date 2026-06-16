"""NCU metric tiers distilled from skills/gpu-kernel-diag/metric_tiers.md.

Source of truth for the tier/pack semantics is the skill file; this module
freezes the concrete metric names so the eval orchestrator can request them
through the KernelGym /evaluate API (which shells out to `ncu --metrics ...`).

Collection policy (LLM-directed hierarchical profiling):
  - first profiled turn: FULL_METRIC_SPACE (T0 + T1 + every T2 pack, one call)
  - later turns: T0 baseline + whatever the analysis LLM requested via its
    `next_metrics` / `packs` JSON decision (validated against this space;
    metric names outside the space are dropped — the skill forbids the LLM
    from inventing metric names).
"""
from __future__ import annotations

# --- T0: cheapest screening set (run for every candidate kernel) -------------
T0_METRICS = [
    "gpu__time_duration.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__throughput.avg.pct_of_peak_sustained_active",
    "lts__throughput.avg.pct_of_peak_sustained_active",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "launch__grid_size",
    "launch__block_size",
    "launch__waves_per_multiprocessor",
]

# --- T1: standard diagnosis set (pattern matching P0-P15) ---------------------
T1_METRICS = [
    # 4 main stalls
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
    "smsp__warp_issue_stalled_wait_per_warp_active.pct",
    # issue / eligibility / IPC
    "smsp__warps_eligible.avg.per_cycle_active",
    "smsp__issue_active.avg.per_cycle_active",
    "sm__inst_executed.avg.per_cycle_active",
    # global access quality + caches
    "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio",
    "l1tex__t_sector_hit_rate.pct",
    "lts__t_sector_hit_rate.pct",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    # the 4 T1 pipes
    "sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_fp64.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active",
    # occupancy limiters (coarse)
    "launch__occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem",
    "launch__occupancy_limit_warps",
    "launch__occupancy_limit_blocks",
    "launch__registers_per_thread",
    "sm__maximum_warps_per_active_cycle_pct",
    # FLOP counters (roofline, KernelAgent-verified set)
    "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_hfma_pred_on.sum",
    "sm__inst_executed_pipe_tensor_op_hmma.sum",
    # branch uniformity (coarse divergence signal)
    "smsp__sass_average_branch_targets_threads_uniform.pct",
]

# --- T2 packs (per-bottleneck fine-grained sets) ------------------------------
_STALL_TYPES = [
    "barrier",
    "branch_resolving",
    "dispatch_stall",
    "drain",
    "imc_miss",
    "lg_throttle",
    "long_scoreboard",
    "math_pipe_throttle",
    "membar",
    "mio_throttle",
    "misc",
    "no_instruction",
    "not_selected",
    "selected",
    "short_scoreboard",
    "sleeping",
    "tex_throttle",
    "wait",
]

_PIPES = ["alu", "fma", "fp16", "fp64", "lsu", "tensor", "xu", "uniform", "cbu", "adu", "tex"]

PACK_SCHED = [
    "smsp__warps_eligible.avg.per_cycle_active",
    "smsp__issue_active.avg.per_cycle_active",
] + [f"smsp__warp_issue_stalled_{s}_per_warp_active.pct" for s in _STALL_TYPES]

PACK_COMPUTE = [
    f"sm__inst_executed_pipe_{p}.avg.pct_of_peak_sustained_active" for p in _PIPES
] + [
    "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_hfma_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_dfma_pred_on.sum",
    "smsp__sass_inst_executed_op_branch.sum",
]

PACK_MEM = [
    # sysmem apertures: UVM / zero-copy leak detector
    "lts__t_sectors_aperture_sysmem_op_read.sum",
    "lts__t_sectors_aperture_sysmem_op_write.sum",
    # instruction mix by memory op (atomics, spill, shared)
    "smsp__sass_inst_executed_op_global_atom.sum",
    "smsp__sass_inst_executed_op_global_red.sum",
    "smsp__sass_inst_executed_op_global_ld.sum",
    "smsp__sass_inst_executed_op_global_st.sum",
    "smsp__sass_inst_executed_op_local_ld.sum",
    "smsp__sass_inst_executed_op_local_st.sum",
    "smsp__sass_inst_executed_op_shared_ld.sum",
    "smsp__sass_inst_executed_op_shared_st.sum",
    # store-side coalescing (T1 only covers ld)
    "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio",
]

PACK_OCC = [
    "launch__occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem",
    "launch__occupancy_limit_warps",
    "launch__occupancy_limit_blocks",
    "launch__registers_per_thread",
    "launch__shared_mem_per_block_static",
    "launch__shared_mem_per_block_dynamic",
    "launch__shared_mem_per_block_driver",
    "sm__maximum_warps_per_active_cycle_pct",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
]

PACK_DIVERGE = [
    "smsp__sass_average_branch_targets_threads_uniform.pct",
    "smsp__thread_inst_executed_per_inst_executed.ratio",
    "smsp__sass_inst_executed_op_branch.sum",
]

PACKS: dict[str, list[str]] = {
    "PACK-MEM": PACK_MEM,
    "PACK-SCHED": PACK_SCHED,
    "PACK-COMPUTE": PACK_COMPUTE,
    "PACK-OCC": PACK_OCC,
    "PACK-DIVERGE": PACK_DIVERGE,
}


def _dedup(seq: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for name in seq:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


FULL_METRIC_SPACE = _dedup(
    T0_METRICS + T1_METRICS + [m for pack in PACKS.values() for m in pack]
)

_ALLOWED = set(FULL_METRIC_SPACE)


def resolve_request(
    next_metrics: list[str] | None,
    packs: list[str] | None,
    *,
    cap: int = 60,
) -> tuple[list[str], list[str]]:
    """Turn the analysis LLM's collection decision into a concrete metric list.

    Returns (metrics, rejected). Always prepends the T0 baseline so successive
    turns stay comparable. Unknown metric names / pack names are rejected
    (the skill forbids the LLM from inventing names). Falls back to the full
    space when nothing valid was requested.
    """
    requested: list[str] = []
    rejected: list[str] = []
    for pack_name in packs or []:
        pack = PACKS.get(str(pack_name).strip().upper())
        if pack is None:
            rejected.append(str(pack_name))
        else:
            requested.extend(pack)
    for name in next_metrics or []:
        name = str(name).strip()
        if name in _ALLOWED:
            requested.append(name)
        elif name:
            rejected.append(name)
    if not requested:
        return list(FULL_METRIC_SPACE), rejected
    metrics = _dedup(T0_METRICS + requested)[: max(cap, len(T0_METRICS))]
    return metrics, rejected
