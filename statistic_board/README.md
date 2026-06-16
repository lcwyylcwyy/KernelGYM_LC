# KernelGYM Statistic Board

A standalone Gradio dashboard for KernelGYM grading outputs.

## Features

- Overview metrics:
  - pass@1
  - best_by_turn_{N} fast@1.2_in_all
  - best_by_turn_{N} fast@1.0_in_all
  - final/correctness_rate
- eval_outputs table:
  - problem_id
  - sample_id
  - eval_status
  - dialogue log preview/path
  - speedup metrics (final and best)

## Run

```bash
cd c:/Users/OT/KernelGYM/statistic_board
pip install -r requirements.txt
python app.py
```

Ubuntu/Linux:

```bash
cd /home/<your_user>/KernelGYM/statistic_board
pip install -r requirements.txt
python app.py
```

Optional arguments:

```bash
python app.py --run-path "C:/Users/OT/Downloads/GKG_Eval_Analysis/drkernel-8b-maxturns3_9060XT_compile/drkernel-8b-maxturns3_9060XT_compile" --host 127.0.0.1 --port 7860
```

Ubuntu/Linux example:

```bash
python app.py --run-path "/home/<your_user>/Downloads/GKG_Eval_Analysis/drkernel-8b-maxturns3_9060XT_compile/drkernel-8b-maxturns3_9060XT_compile" --host 0.0.0.0 --port 7860
```

You can also provide the default path via environment variable:

```bash
export KERNELGYM_RUN_PATH="/home/<your_user>/Downloads/GKG_Eval_Analysis/drkernel-8b-maxturns3_9060XT_compile/drkernel-8b-maxturns3_9060XT_compile"
python app.py
```

Windows PowerShell equivalent:

```powershell
$env:KERNELGYM_RUN_PATH="C:/Users/OT/Downloads/GKG_Eval_Analysis/drkernel-8b-maxturns3_9060XT_compile/drkernel-8b-maxturns3_9060XT_compile"
python app.py
```
prompt analysis:
```
python statistic_board/turn_detail_dashboard.py --results-dir /home/chen/NVS/KernelGYM_LC/drkernel/kernel/scripts/eval/gpt-5.5-weelinking_0526/grading_results --host 172.17.2.155 --port 7860
```

You can pass either:

- The run root path that contains `grading_results`
- Or the direct `grading_results` path
