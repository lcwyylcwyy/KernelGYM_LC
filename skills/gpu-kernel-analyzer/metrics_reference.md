# 指标快速参考（综合版）

> 诊断模式：直接查阈值和结论。教学模式：含物理含义和类比。

---

## Roofline 指标（优化开始前必采，提供上界和停止条件）

### 采集命令
```bash
# 一键采集（推荐）
ncu --section SpeedOfLight_RooflineChart --section SpeedOfLight \
    --csv --log-file roofline.csv ./kernel

# 手动精确采集
ncu --metrics \
  dram__bytes_read.sum,dram__bytes_write.sum,\
  smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,\
  smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,\
  smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,\
  smsp__sass_thread_inst_executed_op_hfma_pred_on.sum,\
  sm__inst_executed_pipe_tensor_op_hmma.sum,\
  gpu__time_duration.sum \
  --csv ./kernel
```

### Roofline 指标含义

> 注：Roofline SOL 采用 KernelAgent 验证的指标组合（来自 meta-pytorch/KernelAgent `ncu_roofline.py`）

| 指标 | 计算方式 | 诊断用途 |
|------|---------|---------|
| `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` | 直接读取 | **Memory SOL**（主指标，包含 L2+DRAM 综合内存利用率） |
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | 直接读取 | **Compute SOL**（SM 计算吞吐，主指标） |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` | >5% = TC 激活 | TC 使用检测 |
| `dram__bytes_read.sum` + `dram__bytes_write.sum` | 直接读取 | 分母：实际 DRAM 流量（计算 AI 用） |
| `gpu__time_duration.sum` | 单位 ns | kernel 执行时间 |
| `smsp__sass_thread_inst_executed_op_ffma_pred_on.sum` | × 2 FLOP | FP32 FMA 计算量 |
| `smsp__sass_thread_inst_executed_op_hfma_pred_on.sum` | × 2 FLOP | FP16 非 TC 计算量 |
| `sm__inst_executed_pipe_tensor_op_hmma.sum` | × 8192 FLOP | TC HMMA 计算量（A100 FP16，warp 粒度）|

⚠️ `dram__throughput`（仅 HBM）和 `gpu__compute_memory_throughput`（L2+HBM 综合）是不同指标。Roofline SOL 用后者更准确。

### 核心计算公式（KernelAgent SOL 方法，无需手算 FLOP）
```python
# 直接用 ncu SOL 指标（推荐，最简洁）
compute_sol = sm__throughput_pct                        # sm__throughput.avg.pct...
memory_sol  = gpu__compute_memory_throughput_pct        # gpu__compute_memory_throughput.avg.pct...
efficiency  = max(compute_sol, memory_sol)              # 主效率指标

# 瓶颈三分类（KernelAgent 标准）
if memory_sol < 60 and compute_sol < 60:  bottleneck = "underutilized"  # 均不饱和→stall/occupancy
elif memory_sol >= compute_sol:            bottleneck = "memory"
else:                                      bottleneck = "compute"

# AI-based（可选，用于定位 Roofline 图上的位置）
total_flop = 2 * M * N * K   # GEMM 解析式
AI = total_flop / (dram_read + dram_write)
```

### 硬件峰值参考

| GPU | FP32 Ridge (F/B) | TC Ridge (F/B) | 显存 BW |
|-----|-----------------|---------------|---------|
| A100 SXM4 | 9.75 | 156 | 2.0 TB/s |
| H100 SXM5 | 20 | 295 | 3.35 TB/s |
| RTX 4090 | 82 | 165 | 1.0 TB/s |

### 停止条件（KernelAgent 标准，95%）
```
efficiency >= 95% → at_roofline = True，停止优化
efficiency 70-95% → 有空间，继续
efficiency < 70%  → 明显不足，进入 SOL 分析

收敛停止：连续 5 轮改善 < 0.1%，也停止（未达 95% 亦然）
```

---

## SOL 四件套（必看）

| 指标 | Section | 诊断阈值 | 物理含义 |
|------|---------|---------|---------|
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | GPU Speed Of Light | >60% → Compute-Bound | SM 计算管道的忙碌程度 |
| `l1tex__throughput.avg.pct_of_peak_sustained_elapsed` | GPU Speed Of Light | >85% → L1/Smem 瓶颈 | L1缓存+Shared Memory复合吞吐 |
| `lts__throughput.avg.pct_of_peak_sustained_elapsed` | GPU Speed Of Light | >85% → L2 瓶颈 | L2缓存（LTS = Level Two Slice）吞吐 |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | GPU Speed Of Light | >70% → DRAM 瓶颈 | 显存（HBM/GDDR）带宽利用率 |

**诊断规则**：sm高+dram低→Compute；dram高→Memory-DRAM；l2高+dram低→Memory-L2；全低→Latency

**教学类比**：这四个是工厂各流水线的"开动率仪表盘"。某条线满载，那条线就是瓶颈。

---

## Compute Workload Analysis

| 指标 | 阈值 | 含义 |
|------|------|------|
| `smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed` | >80% 饱和 | FP32 FMA 管道（FADD/FMUL/FFMA 指令），通用 kernel 最常见瓶颈 |
| `smsp__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` | >80% 饱和 | Tensor Core 管道（WMMA/MMA 指令），GEMM 优化目标 |
| `smsp__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_elapsed` | >80% ⚠️ | FP64（double）管道，数量少、代价高，检查是否误用 double |
| `smsp__pipe_l1tex_cycles_active.avg.pct_of_peak_sustained_elapsed` | >80% | LSU（Load/Store Unit），内存指令密集 |
| `smsp__pipe_alu_cycles_active.avg.pct_of_peak_sustained_elapsed` | >80% | 整数/逻辑运算管道 |
| `smsp__pipe_xu_cycles_active.avg.pct_of_peak_sustained_elapsed` | >80% | SFU（sin/cos/rcp/sqrt），特殊函数 |
| `smsp__inst_executed.avg.per_cycle_active` (IPC) | <1 异常, >3 好 | 每周期平均执行指令数，低→频繁 stall 或 divergence |

**教学补充 - IPC**：IPC 是 CPU 世界的老概念。GPU 的特殊性在于它靠"多 warp 轮换"而不是"深流水线"来提高 IPC。一个 warp stall 时立刻切另一个——所以高 occupancy 是高 IPC 的前提之一。

**教学补充 - Tensor Core**：TC 每周期处理一个 16×16 矩阵乘法 tile，相当于 FP16 模式下 FP32 管道的 16× 吞吐。但它有"对齐要求"——M、N、K 必须是 16 的倍数，否则 CUTLASS/cuBLAS 会退回到 FP32 FMA。

---

## Memory Workload Analysis

### L1 层级

| 指标 | 好 | 中 | 差 | 含义 |
|------|----|----|-----|------|
| `l1tex__t_sector_hit_rate.pct` | >80% | 50-80% | <50% | L1 Cache 扇区命中率 |
| `l1tex__data_bank_conflicts_pipe_lsu.sum` | ~0 | 少量 | >10% of requests | Shared Memory bank conflict 总次数 |
| `l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio` | ~1 | 2-4 | >8 | 每次 Global Load 平均消耗 L1 扇区数（coalescing 指标）|

**Bank Conflict 判断公式**：
```
ratio = l1tex__data_bank_conflicts_pipe_lsu.sum
      / l1tex__t_requests_pipe_lsu_mem_shared_op_ld.sum
> 10% → padding 修复；> 50% → 重新设计布局
```

**教学补充 - Bank Conflict**：Shared Memory 像一栋 32 层楼（32 banks），每层住一排数据。每个 warp 的 32 个线程同时去取数，如果多个线程去同一层楼取不同房间的东西，电梯就得跑多趟（串行化）。修复方法：在数组末尾 pad 一列，让大家去不同楼层。

**教学补充 - Coalescing**：想象 32 个工人同时取货。理想情况是货物排成连续一排，每人取相邻的一件 → 1次行程。最坏情况是货物散落各处 → 32 次单独行程，效率差 32×。

### L2 层级

| 指标 | 好 | 差 | 含义 |
|------|----|----|------|
| `lts__t_sector_hit_rate.pct` | >60% | <30% | L2 Cache 扇区命中率 |
| `lts__t_sectors_op_read.sum` | ≈理论值 | 明显偏高 | L2 读取扇区总数（对比理论 working set）|

### DRAM 层级

| 指标 | 含义 |
|------|------|
| `dram__bytes_read.sum` | 实际读 DRAM 字节数 |
| `dram__bytes_write.sum` | 实际写 DRAM 字节数 |

**诊断用法**：(实际 read + write) / 理论 working set
- <1.2x → 合理；1.2-3x → 有重复加载；>3x → 严重缓存不友好

---

## Occupancy（Latency-Bound 时核心指标）

| 指标 | 阈值 | 含义 |
|------|------|------|
| `sm__warps_active.avg.pct_of_peak_sustained_active` | >75% 好, <25% 差 | 实际 Achieved Occupancy |
| `sm__maximum_warps_per_active_cycle_pct` | — | 理论 Occupancy（由静态参数决定）|
| `launch__occupancy_limit_registers` | 若是最小值 → 限制因素 | 寄存器导致的 occupancy 上限 |
| `launch__occupancy_limit_shared_mem` | 若是最小值 → 限制因素 | Shared memory 导致的上限 |
| `launch__occupancy_limit_warps` | 若是最小值 → 限制因素 | Block size 太小导致的上限 |
| `launch__waves_per_multiprocessor` | 小数 <0.5 → tail effect | 整个 grid 需要多少轮（wave）|

**诊断规则**：Theoretical - Achieved > 20% → 运行时不均衡；<5% → 受静态资源限制，查 limiters

**教学补充 - Occupancy 误区**：Occupancy 高不等于性能好（Compute-Bound kernel 跑满了，再加 warp 也没用）。但 Occupancy 低时一定是危险信号——意味着延迟无法被隐藏，SM 经常空转等数据。

---

## Scheduler Statistics

| 指标 | 阈值 | 含义 |
|------|------|------|
| `smsp__warps_eligible.avg.per_cycle_active` | <1 → 调度器饥渴 | 每周期平均 eligible warp 数 |
| `smsp__issue_active.avg.per_cycle_active` | <0.8 → 有问题 | 每周期实际发射率 |

---

## Warp State Statistics（Stall 根因字典）

查 Details Page → Warp State Statistics，找占比 **>10%** 的 stall：

| Stall | 延迟量级 | 直接含义 | 优化 |
|-------|---------|---------|------|
| `long_scoreboard` | 300-800 cycles | 等 L2/DRAM 数据返回 | ① 增大 occupancy ② shared memory tiling ③ `__ldg()` prefetch |
| `short_scoreboard` | 20-100 cycles | 等 L1/Constant Cache 返回 | 重排指令，增加 RAW 间距 |
| `math_throttle` | — | ✅ Compute 管道满，正常现象 | 升级到 Compute-Bound 分析路径 |
| `wait` | — | 等 `__syncthreads()` barrier | 减少 sync；改 `__syncwarp()`；double buffering |
| `mio_throttle` | — | Memory IO 队列满 | 合并小 load；减少内存指令密度 |
| `lg_throttle` | — | Local/Global 内存队列满 | 同 mio；减少 global/local 访问 |
| `tex_throttle` | — | Texture 单元队列满 | 减少 tex fetch 频率 |
| `membar` | — | 等 `__threadfence()` 完成 | 用 relaxed atomic；减少 fence |
| `no_instruction` | — | 调度器没有 warp 可发射 | 增大 occupancy（最重要！）|
| `not_selected` | — | Warp eligible 但未被选中 | ✅ 正常，说明有足够 warp 竞争 |
| `drain` | — | Warp 完成后退出 | Tail effect，调整 grid size |
| `imc_miss` | — | 立即数常量缓存 miss | 减少每 warp 不同常量的个数 |
| **`barrier`** | — | **等 `__syncthreads` 或 warp group sync** | 减少 sync；`__syncwarp()` 替代；double buffering |
| **`branch_resolving`** | — | **分支条件计算未完成，无法确定跳转目标** | 减少条件分支；预计算条件；排序数据消除 divergence |

> `barrier` 和 `branch_resolving` 来自 KernelAgent `metric_schema.py`，是我们原有 `wait`/`math_throttle` 的更细粒度版本。`barrier` = 跨 warp 同步等待；`branch_resolving` = 分支目标计算本身的延迟。

**教学补充**：把 warp scheduler 想象成一个"叫号系统"。`long_scoreboard` 是去 DRAM 取数的顾客还没回来，`no_instruction` 是等候区彻底没人——后者意味着 occupancy 严重不足，前者意味着需要更多备用顾客来填满等待时间。

---

## Source Counters / Memory Access Patterns

| 指标 | 阈值 | 含义 |
|------|------|------|
| `smsp__thread_inst_executed_per_inst_executed.ratio` | <0.8 → divergence | Thread 效率，反映 branch divergence（传统指标）|
| **`smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct`** | **100%=完美合并，<50%=差** | **Global Load 合并率**（KernelAgent 用法，越高越好） |
| **`smsp__sass_average_branch_targets_threads_uniform.pct`** | **100%=无 divergence，<80%=有问题** | **分支均匀度**（warp 内线程走相同分支的比例）|

> 注：合并率 `smsp__sass_average_data_bytes_per_sector...pct` 是百分比形式（100%=完美），与我们原有的 `l1tex__average_t_sectors_per_request...ratio`（越小越好）互补——前者是 KernelAgent 标准，后者更细粒度。

---

## 原始资源量（Occupancy 深度分析用）

| 指标 | 含义 | 用途 |
|------|------|------|
| **`launch__registers_per_thread`** | 每线程寄存器数（原始值）| 判断是否超过限制（A100 每线程上限 255）|
| **`launch__shared_mem_per_block_allocated`** | 每 block 实际分配的 smem（字节）| 精确计算 smem 限制下的 occupancy 上限 |

```python
# 用原始值验证 occupancy 计算
regs_per_thread = launch__registers_per_thread
smem_per_block  = launch__shared_mem_per_block_allocated  # bytes

# A100 每 SM：65536 寄存器，L1=192KB（可配置 smem 比例）
max_warps_by_regs = 65536 // (regs_per_thread * 32)  # 32 threads/warp
max_warps_by_smem = smem_per_sm_bytes // smem_per_block * warps_per_block
```

---

## 指标组合速查（单指标提示方向，组合锁定根因）

> 完整 16 条 pattern（P0-P15，含联合条件与行动映射）见 gpu-kernel-diag skill 的
> `diag_rules.md` 规则 9。下面是最高频 8 条的速记：

| 组合 | 结论 |
|------|------|
| grid < SM 数 | **其余指标全失真**，先增大 grid 再诊断（P0） |
| mem SOL 高 + sectors/req>4 | 假带宽饱和，是非合并放大流量，先修 coalescing（P2） |
| mem SOL 高 + sectors/req≈1 + L1 hit 低 | 真带宽受限，只能减流量（fp16/fusion/算法），调参无效（P1） |
| SOL 全低 + occ 高 + long_scoreboard 高 | 延迟受限但 warp 已够，**加 occupancy 没用**，要 ILP/pipelining（P5） |
| SOL 全低 + occ 低 + limiter=registers + local_ld>0 | 寄存器溢出双重伤害，先消 spill（P4） |
| compute SOL 高 + tensor pipe<5% + matmul 形 | TC 未使用，最大单项收益 5-20x（P8） |
| tensor pipe 高 + long_scoreboard 高 | TC 等数据，加深流水线 num_stages/TMA（P9） |
| nsys: kernel<100µs × 大量 instances | launch-bound，ncu 指标无意义，先 fusion/CUDA Graph（P11） |
| occ 低 + limiter=registers + **无 spill** + 大 kernel | 可能是策略性高寄存器（大 tile 复用），不一定是病（P15） |

**交叉验证铁律**：任何结论 ≥2 个指标互相印证；修复一个 pattern 后必须重新 profile（瓶颈会转移）。

---

## 架构差异表（阈值与策略按架构调整）

| | A100 (GA100) | H100 (GH100) | B200 (GB100) | RTX 4090 (AD102) |
|---|---|---|---|---|
| SM 数 | 108 | 132 | 148×2 die | 128 |
| smem/SM 上限 | 164 KB | 228 KB | 228 KB | 100 KB |
| L2 | 40 MB | 50 MB | 126 MB | 72 MB |
| HBM/GDDR BW | 2.0 TB/s | 3.35 TB/s | 8 TB/s | 1.0 TB/s |
| 寄存器/SM | 64K×32bit | 64K | 64K | 64K |
| 异步数据通路 | `cp.async` | **TMA**（bulk tensor copy） | TMA + 多播 | `cp.async` |
| MMA 指令代际 | mma (warp 级) | **wgmma**（warp group 级，异步） | **tcgen05**（TMEM 驻留） | mma |
| 新精度 | TF32/BF16 | +FP8 (E4M3/E5M2) | +FP4/FP6 | +FP8 |
| 特有机制 | — | Thread Block Cluster + 分布式 smem | TMEM（Tensor Memory） | — |
| 诊断注意 | baseline | TC 峰值按 wgmma 计，老 mma 路径只能达 ~60% 峰值 | tensor pipe 指标含 tcgen05；TMEM 指标新增 | L2 大，L2 hit 偏高是常态 |

**架构分支规则**：
- 建议 TMA/wgmma/cluster 前先确认 GPU ≥ Hopper（`compute capability ≥ 9.0`），否则给 `cp.async` 替代。
- Triton 在 Hopper+ 会自动用 TMA/wgmma（num_stages 触发），用户无需手写——确认手段是 dump PTX。
- 消费级卡（RTX 系列）：无 NVLink、smem 小、FP64 几乎为零——FP64 pipeline 出现任何占用都是 bug。

---

## 快速诊断索引

```
SOL 看完（gpu__compute_memory_throughput + sm__throughput）→ 确定 bound 类型

efficiency = max(compute_sol, memory_sol)
>= 95%         → 已达 Roofline，停止
60%+ memory    → Memory-Bound → DRAM bytes 超额比 → L2 hit rate → 合并率
60%+ compute   → Compute-Bound → pipeline + IPC + 分支均匀度
两者均 <60%    → Underutilized → Occupancy gap → Limiter → Stall reasons → Tail effect

分类完成后 → 先做组合 pattern 匹配（上表），命中即得根因 + 行动
```

<!-- v2 refine 2026-06-10：新增指标组合速查 + 架构差异表 -->
