from kernelgym.server.api.models import EvaluationRequest
from kernelgym.toolkit.kernelbench.profiling import (
    DEFAULT_NCU_METRICS,
    parse_ncu_csv,
)


def test_evaluation_request_accepts_ncu_profiling_fields():
    request = EvaluationRequest(
        task_id="ncu-test",
        reference_code="class Model:\n    def forward(self, x):\n        return x\n",
        kernel_code="class ModelNew:\n    def forward(self, x):\n        return x\n",
        enable_ncu_profiling=True,
        ncu_metrics=["sm__inst_executed.sum"],
    )

    assert request.enable_ncu_profiling is True
    assert request.ncu_metrics == ["sm__inst_executed.sum"]


def test_parse_ncu_csv_builds_metric_payload_and_scalar_aliases():
    csv_text = "\n".join(
        [
            '==PROF== Connected to process 123',
            '"ID","Process ID","Process Name","Host Name","Kernel Name","Context","Stream","Section Name","Metric Name","Metric Unit","Metric Value"',
            '"1","123","python","host","kernel_a","1","7","NCU","sm__inst_executed_pipe_fma.sum","inst","1,024"',
            '"1","123","python","host","kernel_b","1","7","NCU","sm__inst_executed_pipe_fma.sum","inst","512"',
            '"1","123","python","host","kernel_a","1","7","NCU","sm__inst_executed.sum","inst","4,096"',
            '"1","123","python","host","kernel_b","1","7","NCU","sm__inst_executed.sum","inst","2,048"',
            '"1","123","python","host","kernel_a","1","7","NCU","sm__cycles_active.avg","cycle","50"',
            '"1","123","python","host","kernel_b","1","7","NCU","sm__cycles_active.avg","cycle","70"',
            '"1","123","python","host","kernel_a","1","7","NCU","sm__cycles_elapsed.avg","cycle","100"',
            '"1","123","python","host","kernel_b","1","7","NCU","sm__cycles_elapsed.avg","cycle","120"',
            '"1","123","python","host","kernel_a","1","7","NCU","l1tex__t_sector_hit_rate.pct","%","87.5"',
        ]
    )

    parsed = parse_ncu_csv(csv_text, requested_metrics=DEFAULT_NCU_METRICS)

    assert parsed["status"] == "ok"
    assert parsed["metrics"]["sm__inst_executed_pipe_fma.sum"]["sum"] == 1536.0
    assert parsed["metrics"]["sm__cycles_active.avg"]["avg"] == 60.0
    assert parsed["scalars"]["ncu_sm_inst_executed_pipe_fma_sum"] == 1536.0
    assert parsed["scalars"]["ncu_sm_inst_executed_sum"] == 6144.0
    assert parsed["scalars"]["ncu_l1tex_t_sector_hit_rate_pct"] == 87.5
    assert parsed["scalars"]["ncu_fma_instruction_ratio"] == 0.25
    assert round(parsed["scalars"]["ncu_active_elapsed_cycle_ratio"], 6) == round(60.0 / 110.0, 6)


def test_parse_ncu_raw_page_wide_csv():
    csv_text = "\n".join(
        [
            "==PROF== Connected to process 123",
            '"ID","Process ID","Process Name","Host Name","Kernel Name","Context","Stream","sm__inst_executed.sum","sm__cycles_active.avg"',
            '"","","","","","","","inst","cycle"',
            '"0","123","python","host","kernel_a","1","7","100","20"',
            '"1","123","python","host","kernel_b","1","7","300","40"',
        ]
    )

    parsed = parse_ncu_csv(
        csv_text,
        requested_metrics=["sm__inst_executed.sum", "sm__cycles_active.avg"],
    )

    assert parsed["status"] == "ok"
    assert parsed["metrics"]["sm__inst_executed.sum"]["sum"] == 400.0
    assert parsed["metrics"]["sm__cycles_active.avg"]["avg"] == 30.0
    assert parsed["scalars"]["ncu_sm_inst_executed_sum"] == 400.0
    assert parsed["scalars"]["ncu_sm_cycles_active_avg"] == 30.0
