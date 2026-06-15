# Agent 闭环优化协议（Profile → Diagnose → Prescribe → Patch → Verify → Measure → Reflect）

> 当 LLM/Agent 拥有 GPU 机器 shell 权限时执行本协议。设计参考 KernelAgent（PyTorch, 2026）、
> TritonForge（Meta, 2025）、CudaForge 的已验证实践。
> 核心原则：**一切结论来自硬件指标；一切改动经过数值正确性门禁；一轮只验证一个假设。**

---

## 0. 总流程

```
┌──────────────────────────────────────────────────────────┐
│  baseline 锚定（一次性）                                    │
│  正确性参考 + 基准时间 + baseline profile + Roofline 上界    │
└──────────────┬───────────────────────────────────────────┘
               ↓
┌─→ ① PROFILE   ncu 采集指标（28 指标最小集 / --set full）     │
│   ② DIAGNOSE  指标组合 → BottleneckReport (JSON)            │
│   ③ PRESCRIBE 瓶颈 → 1-3 条候选改动（带 rationale）          │
│   ④ PATCH     只实施 1 条改动（或并行 worker 各实施 1 条）    │
│   ⑤ VERIFY    数值正确性门禁（不过 → 修复或丢弃，不准 benchmark）│
│   ⑥ MEASURE   基准测量（kernel 时间 + 端到端时间）            │
│   ⑦ REFLECT   填写 reflexion JSON → 更新 lessons/avoid 列表  │
└── ⑧ 停止判断：未停止 → 回 ①（带着新 profile 和 lessons）
```

预算：**最多 8 轮**（TritonForge 实证：收益集中在前 3-4 轮）。每轮 wall-clock 目标 < 5 分钟。

---

## 1. Baseline 锚定（进入循环前，一次性完成）

```bash
# 1.1 环境快照（写入工作日志，所有后续轮次复用）
nvidia-smi --query-gpu=name,memory.total,clocks.max.sm --format=csv
python -c "import torch; print(torch.__version__, torch.version.cuda)"

# 1.2 锁频（保证测量可复现；benchmark 与 profile 均在锁频下进行）
sudo nvidia-smi -lgc <base_clock>   # 无 sudo 时跳过，ncu 默认 --clock-control base 已锁
```

```python
# 1.3 正确性参考（golden reference）：永远用最高可用精度的 eager 实现
x = make_inputs(seed=0)
ref_fp64 = reference_op(x.double())          # 黄金参考
ref_same_dtype = reference_op(x)             # 同精度参考（衡量"参考自身误差"用）

# 1.4 baseline 时间（见 §5 测量协议）
t_baseline_kernel, t_baseline_e2e = benchmark(baseline_fn)

# 1.5 baseline profile + Roofline 上界
#     efficiency = max(compute_SOL, memory_SOL)；>=95% 直接终止，不进循环
```

**产物**：`work_log.md` 中的迭代表（每轮一行）：

| 轮 | 改动（一句话） | 正确性 | kernel µs | e2e µs | comp SOL | mem SOL | eff | 决定 |
|----|--------------|--------|-----------|--------|----------|---------|-----|------|
| 0  | baseline     | PASS   |           |        |          |         |     | —    |

---

## 2. PROFILE：指标采集

```bash
# 标准 28 指标最小集（单次采集，开销小；完整清单见 gpu-kernel-analyzer/SKILL.md Step 1 的 ncu --metrics 命令）
ncu --metrics <28_metric_list> --csv --log-file r{N}.csv \
    -k "regex:KERNEL" -c 1 --clock-control base <run_cmd>

# 或完整采集（首轮 / 诊断不明时）
ncu --set full -k "regex:KERNEL" -c 1 -o r{N} <run_cmd>
ncu --import r{N}.ncu-rep --page details > details_{N}.txt   # 先读 rule engine！

# Triton/PyTorch workload：用 NVTX 限定范围，避免 profile 到无关 kernel
ncu -f --nvtx --nvtx-include "BIG_OP/" --set full -o r{N} python bench.py
# Python 侧：with torch.cuda.nvtx.range("BIG_OP"): fn()

# Triton kernel 名定位技巧：kernel 名 = python 函数名（如 regex:matmul_kernel）
# torch.compile 产物：TORCH_LOGS=output_code python ... 可 dump 生成的 triton 代码
```

**注意事项**：
- ncu replay 会保存/恢复 GPU 内存，但**不要**在 ncu 运行里做正确性判断（时钟、缓存状态都不同）。
- kernel < 20µs 时指标噪声大 → 增大问题规模后再 profile。
- 程序内有同名 kernel 多次启动时用 `-c 1 --launch-skip N` 选定稳定实例（跳过 JIT/warmup 阶段）。

**分级采集**：默认只采 T1（28 指标集）。诊断信号模糊时**不要**直接上 `--set full`——
按 `metric_tiers.md` §3 触发表（S1-S11）输出升级 JSON，选择对应 T2 指标包或 T3 手段，
一次性采齐所需包后回到 ③ DIAGNOSE。升级链最多两跳。

---

## 3. DIAGNOSE：结构化诊断（固定 JSON 输出）

按 `diag_rules.md` 规则 1-9 执行（SOL 分类 → 组合 pattern 匹配 → 根因），输出**必须**为以下 JSON
（本地小模型也用此格式，禁止自由文本结论）：

```json
{
  "round": 3,
  "category": "memory | compute | latency | launch_bound | at_roofline",
  "efficiency_pct": 48.9,
  "matched_patterns": ["P2", "P5"],
  "summary": "一句话：瓶颈是什么",
  "root_causes": [
    {
      "cause": "非合并访问放大 DRAM 流量",
      "evidence": [
        {"metric": "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio",
         "value": 7.8, "interpretation": "每请求 7.8 个 sector，理想为 1，流量放大 ~8x"},
        {"metric": "dram__bytes_read.sum", "value": "理论 working set 的 6.2x",
         "interpretation": "与 sectors/req 互相印证"}
      ]
    }
  ]
}
```

**铁律**：每条 root_cause ≥ 2 个互相印证的指标（交叉验证）；单指标异常视为待证假设。

---

## 4. PRESCRIBE + PATCH：处方与实施

```json
{
  "recommended_fixes": [
    {"id": "F1", "fix": "改 SoA 布局使最内层维度连续访问", "expected": "sectors/req 7.8→~1，DRAM 流量 ~÷6",
     "rationale": "对应 P2；OPT-MEM-03", "risk": "low"},
    {"id": "F2", "fix": "num_stages 2→4 隐藏延迟", "expected": "long_scoreboard 38%→<20%",
     "rationale": "对应 P5；注意 smem 限制", "risk": "medium"}
  ],
  "this_round_applies": "F1"
}
```

**实施规则**（来自 KernelAgent 实证教训）：
1. **一轮一个假设**。串行模式只实施 1 条 fix；多 worker 并行时每个 worker 实施不同的 1 条。
2. **禁止复合调参**：不要同时增大 BLOCK_N、BLOCK_K 和 num_stages——三者对 smem/寄存器压力是相乘的，失败后无法归因。
3. **每条 fix 必须写 expected**（预期哪个指标变到多少）——这是 ⑦ Reflect 的判分依据。
4. 局部调参 2-3 轮无果时，允许**架构级重写**（如 split-K↔one-row-per-program、tiling 结构更换）。
   KernelAgent 案例：matvec 卡在 4ms，换 one-row-per-program 架构后 1.95ms。局部最优要靠换结构跳出。
5. 改动前 `cp kernel.py kernel_r{N}.py.bak`——任何回滚都要可一步完成。

---

## 5. VERIFY：数值正确性门禁（不通过不准 benchmark）

### 5.1 测试输入构造（每次验证全部跑）

```python
def make_test_suite(shapes, dtype):
    cases = []
    for seed in (0, 1, 2):                       # 多 seed 随机
        cases.append(randn_inputs(shapes, dtype, seed))
    cases += [
        non_divisible_shapes(),    # 不被 BLOCK 整除的形状（mask 边界 bug 的高发区）
        tiny_shapes(),             # M=N=K=1 / 单元素
        large_values(),            # ±1e4 量级（fp16 溢出检查）
        mixed_sign_denormal(),     # 含负数、接近 0 的值
    ]
    return cases
```

### 5.2 比较标准（按 dtype 分层）

```python
# 黄金参考 = fp64 eager。两层判定：
err_candidate = max_abs_rel_err(candidate_out, ref_fp64)
err_reference = max_abs_rel_err(ref_same_dtype_out, ref_fp64)   # 参考实现自身的舍入误差

# 判定 1（首选，对 fp16/bf16 公平）：候选误差不得明显大于参考自身误差
PASS if err_candidate <= 2.0 * err_reference + eps

# 判定 2（兜底阈值，allclose 风格）：
#   fp32: rtol=1e-4,  atol=1e-5
#   fp16: rtol=1e-2,  atol=1e-3
#   bf16: rtol=2e-2,  atol=1e-2
# reduction 维度很大时（K > 64K）阈值可放宽 ~4x（误差随 √K 增长）
```

### 5.3 附加检查

```bash
# 确定性检查：怀疑 atomic / 归约顺序变化时，连跑两次 bitwise 比较
# （atomicAdd 浮点不满足结合律 → 两次结果不同 = 非确定性，需要明示用户是否可接受）

# CUDA C++ kernel 必跑（Triton 一般可跳过 memcheck）：
compute-sanitizer --tool memcheck  ./kernel    # 越界/未初始化
compute-sanitizer --tool racecheck ./kernel    # smem 竞态（用了手写 smem 同步时）
```

### 5.4 失败处理

```
编译失败  → 读编译器日志 → 修复 → 重新验证（不计入轮次，最多修 3 次后丢弃该 fix）
数值失败  → 按可疑度排查：① 边界 mask ② 归约顺序/精度（累加器应为 fp32）
            ③ 索引/stride 错误 ④ 同步缺失。修不好 → 丢弃该 fix，Reflect 记录 avoid_pattern
精度可疑  → 报告 err_candidate vs err_reference 给用户裁决，绝不静默放行
```

**绝对规则**：正确性失败的 kernel 的任何性能数据**不得**进入迭代表，防止"快但错"污染搜索方向。

---

## 6. MEASURE：基准测量协议

```python
# Triton / PyTorch（首选）：do_bench 自带 L2 缓存清空 + 中位数统计
import triton.testing
t_kernel = triton.testing.do_bench(fn, warmup=25, rep=100, return_mode="median")

# CUDA C++：cuda events，≥3 warmup + ≥10 计时取中位数，期间锁频
# 端到端时间（必须同时测！）：含 host 侧准备（如 transpose/contiguous 预处理）
t_e2e = wall_time(full_op, warmup=3, rep=20, sync=True)
```

**双指标接受判据（Arbiter）**：

```
accept  ⇔  正确性 PASS
        AND t_e2e 改善 ≥ 5%          ← 主判据是端到端，不是 kernel 时间！
        AND t_kernel 没有变差 > 5%

TritonForge 实证教训：host 侧 transpose 让 kernel 指标全面变好、端到端反而慢 4%。
kernel 时间变好但 e2e 变差 → reject，并在 Reflect 中记录"host 开销转移"模式。
```

accept → 新 kernel 成为 current best，进入下一轮；reject → 回滚到上一版本。
（并行 worker 模式：维护 top-K=2 的 beam，各 worker 基于 beam 内 kernel 探索。）

---

## 7. REFLECT：每轮固定填写

```json
{
  "round": 3,
  "was_diagnosis_correct": true,
  "was_fix_effective": false,
  "expected_outcome": "sectors/req 7.8→1",
  "actual_outcome": "sectors/req 7.8→1.1，但 e2e 仅 -2%（出现新瓶颈 long_scoreboard 41%）",
  "lessons": ["合并修复后瓶颈转移到延迟，下一轮处理 occupancy/pipelining"],
  "avoid_patterns": ["不要再调访问布局，已达 ~1"],
  "try_patterns": ["num_stages=3 配合现有 BLOCK，先不动 BLOCK 大小"]
}
```

lessons / avoid / try 三个列表**跨轮累积**，每轮 Prescribe 前必须重读——这是防止 LLM
原地打转（重复生成语义等价变体）的关键机制。

---

## 8. 停止条件（任一满足即停）

```
S1  efficiency = max(compute_SOL, memory_SOL) ≥ 95%        → 已达 roofline
S2  连续 2 轮 reject（无 accept）且无新 try_pattern          → 思路耗尽
S3  累计 8 轮                                                → 预算用尽
S4  连续 3 轮 e2e 改善合计 < 1%                              → 收敛
S5  正确性反复失败同一原因 3 次                               → 上报用户人工介入
```

停止后输出最终报告：迭代表全表 + 最终 vs baseline 的（正确性、e2e 加速比、efficiency、
关键指标 before/after）+ 未尽事项（罗列剩余 try_patterns，供下次继续）。

---

## 9. 端到端模型场景的外层循环（nsys 在外，ncu 在内）

优化对象是整个模型/推理/训练 step 时，先跑外层：

```bash
nsys profile --stats=true -t cuda,nvtx,osrt -o sys python train.py   # 限定 2-10 个 step
nsys stats sys.nsys-rep --report cuda_gpu_kern_sum                    # 热点 kernel 表
nsys analyze sys.nsys-rep                                             # expert system 规则
```

```
外层决策：
  GPU 利用率 < 80%（gaps/同步/dataloader） → 先修系统问题（diag_rules.md 规则 5/8），
                                              kernel 级优化在此之前都是浪费
  单 kernel Time% > 30%                     → 该 kernel 进入 §0 内层闭环
  大量 <100µs 小 kernel                     → 先 fusion / CUDA Graph，再看剩余热点
每完成一轮内层闭环 → 重跑 nsys 确认端到端收益落袋（kernel 加速 ≠ 模型加速）。
```

---

## 10. 本地小模型（7B-70B）降级模式

能力受限的模型执行同一闭环，但做三件简化：
1. **只用 28 指标最小集 + diag_rules.md 的 if-then 规则表**，禁止自由推理诊断；
   匹配不到 pattern 时输出 `"category": "unknown"` 并请求人工。
2. **只从 optimization_db.md / 规则 10（Triton 表）选现成 fix**，禁止发明新优化。
3. 所有阶段输出严格按本文件 JSON schema，由外层脚本校验字段完整性后才执行下一步。
