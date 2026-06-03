"""Default Nsight Compute metric configuration."""

DEFAULT_NCU_METRICS = [
    "sm__inst_executed_pipe_fma.sum",
    "sm__inst_executed.sum",
    "sm__cycles_active.avg",
    "sm__cycles_elapsed.avg",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "l1tex__t_sector_hit_rate.pct",
    "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
    "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
]
