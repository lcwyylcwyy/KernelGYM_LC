# 诊断规则手册

> gpu-kernel-diag 速效诊断的完整判断规则，严格按此执行

---

## 规则 -1：黄金法则

**Profile → Diagnose → Plan，严格按此顺序。绝不在看数据之前就给建议。**

---

## 规则 0：数据质量检查

收到数据后，先检查：
- 时钟是否锁定（ncu 默认锁定 SM clock，nsys 不锁）
- Kernel 执行时间 < 20µs？→ 数据可能不稳定，提示用户增大 workload
- 是否有 multi-pass 导致的不一致（hit rate 异常 >100% 或 <0%）→ 提示可能是 replay 误差

---

## 规则 0.5：先读 NCU Rule Engine（必做，5 秒找答案）

```bash
# 从已有的 .ncu-rep 文件导出 details 页面
ncu --import profile.ncu-rep --page details > details.txt
```

details 页面包含 NCU 内置规则引擎的输出，格式如下：
```
OPT   Est. Speedup: 24.5%
      On average, each warp spends X cycles stalled waiting for a scoreboard
      dependency on a L1TEX operation. Find the instruction producing the data...

OPT   Est. Speedup: 10.8%
      The memory access pattern for global loads from L1TEX might not be optimal.
      On average, only 7.6 of the 32 bytes transmitted per sector are utilized...
```

**规则**：按 `Est. Speedup %` 降序排列，优先解决最大的那个。NCU 规则引擎往往直接指向答案，不要绕过它。

---

## 规则 1：SOL 分类阈值

> **指标区分**（不要混用）：
> - `dram__throughput` = HBM 带宽 SOL → 用于本节的 DRAM 层分类
> - `gpu__compute_memory_throughput` = L2+DRAM 综合 SOL → 用于规则 6D Roofline 效率计算
> 
> Memory-bound kernel 的 `gpu__compute_memory_throughput` 通常高于 `dram__throughput`，因为它还含 L2 流量。

```python
def classify_bottleneck(sm_pct, l1_pct, l2_pct, dram_pct):
    
    # COMPUTE-BOUND
    if sm_pct > 60 and dram_pct < 50 and l2_pct < 70:
        return "COMPUTE-BOUND"
    
    # MEMORY-BOUND（DRAM带宽饱和）
    if dram_pct > 70:
        return "MEMORY-BOUND-DRAM"
    
    # MEMORY-BOUND（L2饱和）
    if l2_pct > 85 and dram_pct < 60:
        return "MEMORY-BOUND-L2"
    
    # MEMORY-BOUND（L1/Shared Memory饱和）
    if l1_pct > 85 and l2_pct < 60:
        return "MEMORY-BOUND-L1"
    
    # LATENCY / OCCUPANCY 不足
    if sm_pct < 40 and l1_pct < 40 and l2_pct < 40 and dram_pct < 40:
        return "LATENCY-BOUND"
    
    # MIXED（多个同时饱和，较少见）
    return "MIXED"
```

---

## 规则 2：Compute-Bound 子分析

**Step 0（先于 Pipeline 检查）：TC 是否被使用？**

```python
tc_pct = smsp__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active

if sm_pct > 60 and tc_pct < 5:
    # Pattern F：SM 忙碌，但 TC 完全未使用
    # 典型于：矩阵乘法 / Attention / Convolution 用 FP32 FMA 实现
    警告：Tensor Core 未使用，潜在加速 5-20x
    → 检查数据类型（FP16/BF16 才能走 TC）
    → 改用 wmma API / mma PTX / cuBLAS / cutlass
    → 优先级：最高，先于其他 Pipeline 优化
```

**Step A：确定饱和 Pipeline**

| 指标 | 阈值 | 含义 |
|------|------|------|
| `smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed` | >80% | FP32 FMA 饱和 |
| `smsp__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` | >80% | Tensor Core 饱和（正常，说明 TC 利用好） |
| `smsp__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_elapsed` | >80% | FP64 饱和（确认是否必须用 double？） |
| `smsp__pipe_l1tex_cycles_active.avg.pct_of_peak_sustained_elapsed` | >80% | LSU 内存指令密集（Compute-Bound 但被 mem 指令拉住）|

**Step B：IPC 检查**

`smsp__inst_executed.avg.per_cycle_active`（IPC）
- A100 理论最大 IPC ≈ 4（每 SMSP 每周期发射 1 条，4 个 SMSP）
- IPC < 2 → 严重的指令级并行不足或频繁 stall
- IPC 接近理论 → kernel 已经很高效

**Step C：Branch Divergence 检查**

`smsp__thread_inst_executed_per_inst_executed.ratio`
- 理想 = 1.0（所有 32 个线程执行每条指令）
- < 0.5 → 严重 divergence，一半以上线程被 mask 掉
- 优化方向：重排数据让同一 warp 的线程走相同分支

---

## 规则 3：Memory-Bound 子分析

### 3A：L1 层级分析

| 指标 | 好 | 中 | 差 |
|------|----|----|-----|
| `l1tex__t_sector_hit_rate.pct` | >80% | 50-80% | <50% |
| `l1tex__data_bank_conflicts_pipe_lsu.sum` | 0 | 少量 | >10% of requests |
| `l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio` | ~1 | 2-4 | >8 |
| `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct` | ~100% | 50-80% | <25% |

**Bank Conflict 判断**：
```
bank_conflict_ratio = l1tex__data_bank_conflicts_pipe_lsu.sum 
                    / l1tex__t_requests_pipe_lsu_mem_shared_op_ld.sum

> 10%  → 显著 bank conflict → 需要 padding
> 50%  → 严重 bank conflict → 重新设计 shared memory 布局
```

**Coalescing 指标说明（两种表示方式，含义相同，选一个用）**：
```
方式 A（Sectors/Request，越低越好）：
  l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio
  理想：1.0（一次请求只需 1 个 32-byte sector，完美合并）
  2-4：轻微 strided；>8：严重不连续访问
  公式：sectors_issued / load_requests

方式 B（Bytes Utilized/Sector，越高越好）[KDA/KernelAgent 标准]：
  smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct
  理想：100%（每个 sector 的所有 32 bytes 都被使用）
  50-80%：中等浪费；<25%：严重浪费，等效于方式 A 的 ratio > 8

换算关系：bytes_pct ≈ 100% / sectors_per_request
```

### 3B：L2 层级分析

| 指标 | 好 | 差 |
|------|----|----|
| `lts__t_sector_hit_rate.pct` | >60% | <30% |
| `lts__t_sectors_op_read.sum` (相对 working set) | ≈理论值 | 明显高于理论 |

L2 miss rate 高 → working set 超过 L2 容量，需要：
- 分块（tiling）让数据适配 L1/Shared Memory
- 减少每次 kernel 需要的数据量

### 3C：DRAM 层级分析

```
理论 working set = 输入 + 输出 tensor 大小
实际 DRAM reads = dram__bytes_read.sum
超额比 = 实际/理论

超额 < 1.2x → 基本合理
超额 1.2-3x → 有重复加载，改善 L2/L1 复用
超额 > 3x → 严重的缓存不友好，需要根本性重构
```

---

## 规则 4：Latency/Occupancy 子分析

### 4A：Occupancy 计算

```
achieved_occ = sm__warps_active.avg.pct_of_peak_sustained_active  (%)
theoretical_occ = sm__maximum_warps_per_active_cycle_pct (%)

gap = theoretical_occ - achieved_occ

gap > 20%  → 运行时负载不均衡（某些 SM 闲置）
gap < 5%   → occupancy 限制因素是理论限制，看 limiters
```

### 4B：Occupancy Limiter 判断

查看 `launch__occupancy_limit_*` 中最小的那个：

| Limiter | 指标 | 优化方向 |
|---------|------|----------|
| 寄存器 | `launch__occupancy_limit_registers` | 减少变量，用 `__launch_bounds__` |
| Shared Memory | `launch__occupancy_limit_shared_mem` | 减少 smem 或动态分配 |
| Block 大小 | `launch__occupancy_limit_warps` | 增大 block size（最小 128 threads）|
| Block 数/SM | `launch__occupancy_limit_blocks` | 一般无需优化 |
| Barriers | `launch__occupancy_limit_barriers` | 减少 `barrier.sync` / `mbarrier` 用量（Hopper+ TMA kernel 易出现）|

### 4C：Warp Stall 原因解析

| Stall 原因 | 直接含义 | 优化方向 |
|-----------|---------|---------|
| `long_scoreboard` | 等 L2/DRAM 返回数据 | 增大 occupancy 隐藏延迟；用 prefetch（`__ldg`）；shared memory tiling |
| `short_scoreboard` | 等 L1/Constant Cache 返回 | 增加指令级并行；重排指令顺序 |
| `math_throttle` | Compute 流水线满载 | ✅ 正常现象（kernel 是 compute-bound） |
| `wait` | 等 `__syncthreads` barrier | 减少 sync 次数；改用 `__syncwarp`（warp 内） |
| `mio_throttle` | MIO（memory IO）队列满 | 减少内存指令密度；merge 多个小 load 为大 load |
| `lg_throttle` | L/G 内存管道节流 | 同 mio_throttle；减少 global/local 内存指令 |
| `tex_throttle` | Texture 单元队列满 | 减少纹理访问频率 |
| `membar` | 内存栅栏等待 | 减少 `__threadfence()`；用 relaxed atomic |
| `no_instruction` | 调度器没有 warp 可发射 | 增大 occupancy（block size/减少资源占用） |
| `not_selected` | Warp 就绪但没被选中 | 正常，有足够 warp 时自然缓解 |
| `drain` | Warp 完成退出 | Tail effect，增大 grid size |
| `imc_miss` | 立即数常量缓存 miss | 减少每个 warp 用到的不同常量个数 |
| `short_scoreboard` (dcc/idc/imc) | 各类常量缓存 miss | 检查常量内存访问模式 |
| **`barrier`** | **等 `__syncthreads` / warp group barrier** | **减少 sync 点；改 `__syncwarp()`；double buffering** |
| **`branch_resolving`** | **分支跳转目标未确定** | **减少条件分支；预计算条件；数据排序消除 divergence** |

**判断优先级**：只关注占比 >10% 的 stall 原因，其余忽略。
占比最高的那个就是根因。

### 4D：Tail Effect 检查

```
waves = launch__waves_per_multiprocessor

< 1.0 → 只有一波，SM 被均分，问题不大
1.0-2.0 → 两波，第二波 SM 较少，轻微 tail effect  
> 2.0 → 多波，关注最后一波是否很小
```

最后一波利用率 = `(grid_size % (SM_count × blocks_per_SM)) / (SM_count × blocks_per_SM)`  
< 50% → 明显 tail effect → 调整 grid size 或 persistent kernel

**可变长度 Tail Effect（更危险）**：batch 中每个 CTA 的工作量不同时，最慢的 CTA 决定整体延迟。
信号：`--page details` 提示 "One or more SMs have a much lower number of active cycles than average"，或 PM timeline 尾部呈**渐变下滑**（非突降）。
修复：按长度排序输入（sorted batching）；Split-K 分解长序列；chunked kernel。

---

## 规则 5：Nsight Systems 时间线规则

| 现象 | 诊断 | 优化 |
|------|------|------|
| GPU timeline 有大段空白 | CPU 没有及时 submit 工作 | 用 CUDA Graph；减少 API 调用 |
| cudaMemcpy 占比 >20% | 数据传输瓶颈 | Pinned memory + async copy + overlap |
| 多个 stream 但 kernel 顺序 | Stream 有隐式依赖 | 检查 event 和 sync 语义 |
| cudaMalloc 频繁出现 | 动态分配开销 | 预分配 memory pool（cnmem/rapids-rmm）|
| NCCL AllReduce 占 >30% | 通信瓶颈 | 启用 gradient compression；增大 batch |
| CPU 线程占满但 GPU 利用低 | DataLoader CPU 瓶颈 | 增加 num_workers；用 DALI |

---

## 规则 6：Roofline 上界分析（优化开始前必做，提供停止条件）

### 6A：采集命令

```bash
# 推荐：ncu 内置 Roofline section
ncu --section SpeedOfLight_RooflineChart \
    --section SpeedOfLight \
    --csv --log-file roofline.csv ./kernel

# 手动精确采集（KernelAgent 验证的指标集）
ncu --metrics \
  sm__throughput.avg.pct_of_peak_sustained_elapsed,\
  gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,\
  sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,\
  dram__bytes_read.sum,dram__bytes_write.sum,\
  smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,\
  smsp__sass_thread_inst_executed_op_hfma_pred_on.sum,\
  sm__inst_executed_pipe_tensor_op_hmma.sum,\
  gpu__time_duration.sum \
  --csv --log-file roofline_raw.csv ./kernel
```

> 关键：Memory SOL 用 `gpu__compute_memory_throughput`（L2+DRAM 综合），不是单纯的 `dram__throughput`（仅 HBM）。这是 KernelAgent `ncu_roofline.py` 的验证标准。

### 6B：计算 Arithmetic Intensity

```python
# ---- 方法 A：算法已知（首选，精确）----
# GEMM (M×N×K)：每个乘加 = 2 FLOP
total_flop = 2 * M * N * K

# Conv2D (N,C,H,W → N,K,P,Q，filter R×S)：
total_flop = 2 * N * K * P * Q * C * R * S

# ---- 方法 B：从 ncu 指令计数推算（通用 kernel）----
# FP32（每条指令作用于整个 warp，×32 threads）
flop_fp32 = (ffma_sum * 2 + fadd_sum + fmul_sum)

# FP16 非 TC 路径
flop_fp16 = hfma_sum * 2

# Tensor Core HMMA（A100 FP16，16×16×16 tile per warp instr = 8192 FLOP）
flop_tc = hmma_warp_instr_sum * 8192

total_flop = flop_fp32 + flop_fp16 + flop_tc

# Arithmetic Intensity
dram_bytes = dram__bytes_read_sum + dram__bytes_write_sum
AI = total_flop / dram_bytes   # 单位：FLOP/byte
```

### 6C：硬件峰值参考表

| GPU | FP32（非TC） | BF16/FP16 TC | 显存 BW | Ridge(FP32) | Ridge(TC) |
|-----|------------|-------------|---------|------------|----------|
| A100 SXM4 | 19.5 TFLOPS | 312 TFLOPS | 2.0 TB/s | 9.75 F/B | 156 F/B |
| H100 SXM5 | 67 TFLOPS | 989 TFLOPS | 3.35 TB/s | 20 F/B | 295 F/B |
| RTX 4090 | 82.6 TFLOPS | 165 TFLOPS | 1.0 TB/s | 82 F/B | 165 F/B |
| RTX 3090 | 35.6 TFLOPS | 71 TFLOPS | 0.94 TB/s | 38 F/B | 75 F/B |

### 6D：效率计算与定位（KernelAgent SOL 方法）

```python
# 方法一：直接用 SOL 指标（推荐，无需手算 FLOP）
compute_sol = sm__throughput_pct           # sm__throughput.avg.pct_of_peak_sustained_elapsed
memory_sol  = gpu__compute_memory_pct      # gpu__compute_memory_throughput.avg.pct...
efficiency  = max(compute_sol, memory_sol) # 主效率指标（0-100）

uses_tc = sm__pipe_tensor_cycles_active_pct > 5.0

# 瓶颈三分类
if memory_sol < 60 and compute_sol < 60:  bottleneck = "underutilized"
elif memory_sol >= compute_sol:            bottleneck = "memory"
else:                                      bottleneck = "compute"

# 方法二：AI-based（可选，用于 Roofline 图定位）
AI = total_flop / (dram_read + dram_write)
# AI vs Ridge Point → 确认 memory/compute bound 与 SOL 结论一致
```

**定位判断**：
```
efficiency >= 95%   → at_roofline，停止优化
memory_sol >= 60%   → Memory-Bound
compute_sol >= 60%  → Compute-Bound
两者均 < 60%        → Underutilized（stall/occupancy 问题，不是 BW 或计算饱和）
```

### 6E：停止条件（KernelAgent 标准）

```
efficiency >= 95%  → at_roofline = True，停止（主条件）
efficiency < 95%   → 继续，但检查收敛：
  连续 5 轮改善 < 0.1%  → 也停止（已收敛，即使未达 95%）
efficiency < 50%   → 有明显空间，优先解决当前瓶颈
```

### 6F：迭代效率追踪表（每轮填写）

| 版本 | 描述 | GFLOPS | AI (F/B) | Roof | Efficiency | 主要改动 |
|------|------|--------|---------|------|------------|---------|
| v0 | baseline | | | | | — |
| v1 | +smem tiling | | | | | |
| v2 | +software pipeline | | | | | |

efficiency 停止增长即为收敛信号。

---

## 规则 7：Source-Level 精确定位（per-line stall 归因）

当规则 0.5/3/4 定位了 stall 类型但不知道具体是哪条指令时，收集 source-level 数据。

### 7A：采集命令

```bash
# 编译时必须加 -lineinfo，否则 source view 为空
nvcc -lineinfo -O2 -o kernel kernel.cu

# 两次采集：full overview + source-level per-PC
ncu --set full --section PmSampling --section PmSampling_WarpStates \
    -k "regex:KERNEL_NAME" -c 1 -o full_profile ./kernel

ncu --set source --section SourceCounters \
    -k "regex:KERNEL_NAME" -c 1 -o source_profile ./kernel

# 导出 details 页面（NCU rule engine）
ncu --import full_profile.ncu-rep --page details > details.txt
```

> **注**：`PmSampling` 是 PM 采样时间线，不在 `--set full` 中，需单独加。

### 7B：PM 采样时间线形态解读

PM 采样给出 SM 吞吐率随时间的变化。四种典型形态：

| 形态 | 描述 | 含义 |
|------|------|------|
| **平顶 → 陡降** | 高利用率 → 突然归零 | ✅ 理想：均衡、Grid 充足 |
| **平顶 → 渐降（长尾）** | 末尾逐渐下滑 | ⚠️ 可变长度 Tail Effect（部分 CTA 很重） |
| **持续低位** | 全程利用率低 | ❌ Grid 太小（blocks < SM 数）或严重 stall |
| **锯齿波动** | 高低交替 | ⚠️ 无 compute-memory overlap（单缓冲，应用 double buffering）|

### 7C：Source-level hotspot 提取

```bash
# ncu 导出 source 页面（含每行 stall 计数）
ncu --import source_profile.ncu-rep --page source > source.txt

# 重点关注
# 1. 哪几行有最多的 long_scoreboard 样本 → 那条 LDG/LDS 是根因
# 2. 哪几行有最多的 barrier 样本 → 那个 __syncthreads() 是瓶颈
# 3. 哪几行有最多的 short_scoreboard 样本 → 依赖链太短，需重排指令
```

**判断**：只关注 stall 样本占比 >10% 的代码行，其余忽略。找到热点行后直接对应 `optimization_db.md` 中的优化策略。

---

## 规则 8：Nsight Systems Expert System（系统级分析优先）

NSYS 有内置专家系统，等价于 NCU 的 Rule Engine，先运行再做手动分析。

### 8A：采集 + 自动规则检测

```bash
# 采集（--stats=true 同时输出热点 kernel 表）
nsys profile --stats=true -o report ./your_app

# 运行 Expert System（6 条自动规则）
nsys analyze report.nsys-rep
```

### 8B：Expert System 六条规则

| 规则 | 检测内容 | 修复方向 |
|------|---------|---------|
| `cuda_memcpy_async` | Async memcpy 但内存是 pageable → 实际退化为同步 | 用 `cudaHostAlloc` / `cudaMallocHost` 分配 pinned memory |
| `cuda_memcpy_sync` | 同步 `cudaMemcpy` 阻塞 CPU thread | 改为 `cudaMemcpyAsync` + `cudaEventSynchronize` |
| `cuda_memset_sync` | 同步 `cudaMemset` 阻塞 CPU thread | 改为 `cudaMemsetAsync` |
| `cuda_api_sync` | `cudaDeviceSynchronize` / `cudaStreamSynchronize` 过多 | 改用 `cudaEventSynchronize`；用 CUDA Graph 消除 |
| `gpu_gaps` | GPU idle > 500ms（可调阈值） | 用 CPU sampling / OS backtrace 找阻塞原因 |
| `gpu_time_util` | GPU 时间利用率低（按时段检测） | 同 gpu_gaps，但检测低效区间而非完全空闲 |

### 8C：`nsys stats` 热点 Kernel 定位

```bash
# 找 Total Time 最高的 kernel（最重要的单条命令）
nsys stats report.nsys-rep --report cuda_gpu_kern_sum

# 内存操作耗时排名
nsys stats report.nsys-rep --report cuda_gpu_mem_time_sum

# CUDA API + kernel 总览（看 API 开销是否过大）
nsys stats report.nsys-rep --report cuda_api_gpu_sum

# OS 运行时 API（线程同步、文件 IO 等）
nsys stats report.nsys-rep --report osrt_sum
```

`cuda_gpu_kern_sum` 输出关键字段：

| 字段 | 含义 |
|------|------|
| `Time%` | 占所有 kernel 总时间的百分比 → 优先优化这个 |
| `Total Time` | 所有实例总耗时 |
| `Instances` | 调用次数 |
| `Avg / StdDev` | 平均时间 / 标准差高 → 执行时间不稳定 |

### 8D：Queue Time 诊断（CPU-GPU 协作健康度）

```bash
# 查看 kernel launch 的各阶段延迟
nsys stats report.nsys-rep --report cuda_kern_exec_sum
```

关键字段：
- **API Time**：CPU 执行 launch 调用的耗时
- **Queue Time**（QAvg）：launch 返回 → kernel 开始执行 的等待时间
  - Queue Time 高 → GPU 繁忙排队，CPU 发射 < GPU 消费速度（正常）
  - Queue Time ≈ 0 → GPU 空等 CPU → CPU 是瓶颈
- **Kernel Time**（KAvg）：kernel 在 GPU 上的实际执行时间

### 8E：高级 Recipe 分析

```bash
# Kernel 执行时间稳定性（是否每次一样？）
nsys recipe cuda_gpu_kern_pace --input report.nsys-rep --output pace_dir

# GPU 利用率热力图（时间×GPU 二维图）
nsys recipe cuda_gpu_time_util_map --input report.nsys-rep --output heatmap_dir

# 精确找 GPU idle gap（含推测根因）
nsys recipe gpu_gaps --input report.nsys-rep --output gaps_dir

# 优化前后对比
nsys recipe diff --input before.nsys-rep after.nsys-rep --output diff_dir

# NCCL 通信 vs 计算 overlap 分析
nsys recipe nccl_gpu_overlap_trace --input report.nsys-rep --output nccl_dir
```

---

## 规则 9：指标组合 Pattern 库（多指标联合判断，诊断的核心）

> 单个指标只能提示方向，**组合 pattern 才能锁定根因**。按编号顺序匹配，命中即输出
> pattern ID + 证据。多个命中时，编号靠前者优先处理（已按因果上游排序）。
> 缩写：cSOL=`sm__throughput`，mSOL=`gpu__compute_memory_throughput`，
> occ=`sm__warps_active`(achieved)，s/r=`sectors_per_request`(global ld)，
> stall_X=`smsp__warp_issue_stalled_X_per_warp_active.pct`

| ID | 联合条件（全部满足才命中） | 诊断 | 行动（对应 OPT-* / Triton 表） |
|----|--------------------------|------|------------------------------|
| **P0** | `launch__grid_size` < SM 数 或 waves < 0.6 | Grid 不足，**其余指标全部失真** | 先增大 grid（减小 BLOCK / Split-K），重新 profile 后再诊断 |
| **P1** | mSOL>70 + s/r≈1 + L1 hit<30 + DRAM bytes ≈ 理论 working set | 真·带宽受限（流式访问，无浪费） | 减流量：fp16/bf16 **输入与累加**、kernel fusion、算法提高 AI。调 occupancy/block 无效。🛑 **绝不把最终输出 tensor 降到 fp16/bf16**——见下方红线 |

> 🛑 **KernelBench correctness 红线（减流量时必读）**：最终写出的输出 tensor **dtype 必须 fp32**。grader `correctness.py:92` 的 `torch.allclose(out, out_new, atol=1e-2,rtol=1e-2)` **不做 dtype cast**，fp32-ref 对比 fp16-out 直接抛 `RuntimeError: Float did not match Half`，整轮归零（实测 P41 ncu-mode 因此 7/14 轮报废）。"fp16 输出减半写流量"是致命诱饵。降精度只用在 conv/GEMM 输入 + tensor-core 累加路径；输出 `empty_like(x, dtype=torch.float32)`。
| **P2** | mSOL>60 + s/r>4（或 bytes/sector 利用率<50%） | 非合并访问**放大**流量，带宽是假饱和 | OPT-MEM-03（SoA/连续化）。修复优先级高于一切——其他指标都被它污染 |
| **P3** | mSOL>60 + L2 hit>70 + dram SOL<50 + l1tex 或 lts SOL>85 | L2/L1 带宽瓶颈（数据在片上但复用结构差） | OPT-MEM-01 tiling 提高 L1/smem 复用；增大 BLOCK 提高每 tile 复用 |
| **P4** | cSOL<40 + mSOL<40 + occ<40 + limiter=registers + `local_ld>0` | 寄存器溢出双重伤害（occupancy 低 + spill 流量） | OPT-SPILL-01。先消 spill 再谈其他 |
| **P5** | cSOL<40 + mSOL<40 + occ>60 + stall_long_scoreboard>30 | 延迟受限但 warp 已够——**加 occupancy 没用** | 提高 ILP：software pipelining、num_stages↑、向量化 load、每线程多元素 |
| **P6** | cSOL<40 + mSOL<40 + occ<40 + limiter=shared_mem | smem 限制并发 | OPT-OCC-03；Triton：num_stages↓ 或 BLOCK↓ |
| **P7** | occ 正常 + `warps_eligible<1` + `issue_active<0.6` + stall_barrier(或 wait)>15 | 同步串行化（所有 warp 同时卡在 barrier） | OPT-SYNC-01 double buffering；减少 sync 点 |
| **P8** | cSOL>60 + tensor pipe<5% + workload 是 matmul/conv/attention 形 | **Tensor Core 未使用**（最大单项收益，5-20x） | 改 fp16/bf16 + wmma/mma/`tl.dot`/cuBLAS。形状对齐 16 的倍数 |
| **P9** | tensor pipe>50 + mSOL>60 + stall_long_scoreboard>20 | TC 在等数据（喂不饱） | 数据管线：num_stages↑、TMA(Hopper+)、double buffering、增大 BLOCK_K |
| **P10** | cSOL>60 + IPC<1.5 + branch_uniform<80%（或 thread_inst ratio<0.8） | Branch divergence 浪费发射槽 | 3A-DIV：数据重排/predication/warp 对齐分支 |
| **P11** | nsys: kernel avg<100µs + instances 高 + GPU gaps 多（或 QTime≈0） | Launch-bound——**ncu 指标无意义** | OPT-COMP-02 fusion / OPT-GRAPH-01 CUDA Graph。别浪费时间调 kernel 内部 |
| **P12** | fp64 pipe>0 且设计上全 FP32 | 字面量/库函数误用 double | OPT-FP64-01 |
| **P13** | theoretical_occ − achieved_occ > 20% + SM active cycles max/min > 2x | 负载不均衡 / 可变长度 tail | OPT-TAIL-01（sorted batching / split-K / persistent） |
| **P14** | cSOL 与 mSOL 都在 55-75 区间且接近 | 计算访存交错、互相等待（多见于未 pipeline 的 tiled kernel） | Overlap：async copy（`cp.async`/TMA）、num_stages≥3，让 load 与 MMA 并行 |
| **P15** | occ<25 + limiter=registers + **无** spill + kernel 本身大（如 attention/GEMM） | 高寄存器是策略性的，**不一定是病** | 先看 stall：若 long_scoreboard 低则保持现状（大 tile 高复用 > 高 occupancy）；高才考虑减 BLOCK |

**使用规则**：
1. 命中多个 → 按 P0 > P11 > P2 > P12 > P8 > 其余 的优先级处理（系统级和"指标污染源"在前）。
2. 修复一个 pattern 后**必须重新 profile**——瓶颈会转移（P2 修完常变 P5）。
3. 任何 pattern 都需 ≥2 个指标同时满足才算命中——这就是交叉验证。
4. 全部不命中且 efficiency<60 → 走规则 7 source-level 定位，不要猜。

---

## 规则 10：Triton Kernel 参数映射表（指标 → meta-parameter 行动）

> Triton 把 CUDA 的优化自由度压缩为少数 meta-parameters。诊断到的瓶颈按此表映射为参数调整。
> **复合压力警告（KernelAgent 实证）**：BLOCK_M/N/K、num_stages、num_warps 对
> smem ≈ (BLOCK_M+BLOCK_N)×BLOCK_K×dtype×num_stages 和寄存器是**相乘**关系，一次只动一个。

| 诊断（pattern/指标） | Triton 行动 | 说明 |
|---------------------|------------|------|
| P5 / stall_long_scoreboard 高 | `num_stages` 2→3→4 | 编译器自动生成 cp.async/TMA 流水线；smem 随之线性涨，盯 P6 |
| P6 / limiter=shared_mem | `num_stages`↓ 或 BLOCK_K↓ | H100 smem 228KB/SM 上限；留 >2 CTA/SM 的余量 |
| P4/P15 / limiter=registers | `num_warps`↓（8→4）或 BLOCK↓；极端时 `maxnreg` | num_warps 大→每 warp 可用寄存器少→易 spill |
| P0 / grid 太小 | BLOCK_M/N↓ 或加 `GROUP_M` split / Split-K | grid = cdiv(M,BM)×cdiv(N,BN)，至少 ≥ 2×SM 数 |
| P2 / s/r 高 | 检查指针算术：连续维度必须由 `tl.arange` 的**最后一维**索引；输入 `.contiguous()` | Triton 不会替你修布局 |
| P8 / TC 未用 | 用 `tl.dot`（勿手写乘加循环）；操作数 fp16/bf16；BLOCK_M/N/K ≥16 且为 16 倍数 | fp32 输入加 `allow_tf32=True` 走 TF32 TC |
| P9 / TC 等数据 | BLOCK_K↑（64→128）+ `num_stages`≥3 | 提高每次同步搬运的计算量 |
| P13 / 变长 tail | persistent kernel 模式（`tl.program_id` 循环领任务）或宿主侧 sorted batching | |
| P1 / 真带宽受限 | 输出/中间量降精度；fuse 上下游 op 进同一 kernel | 调参无效，要改算法结构 |
| 吞吐型 elementwise | `BLOCK_SIZE` 512-4096 + autotune | 小 BLOCK 是常见低级错误（TritonForge 95x 案例：BLOCK=4） |
| 不确定最优组合 | `@triton.autotune` 网格：BLOCK ∈ {32..256}, num_warps ∈ {2,4,8}, num_stages ∈ {2,3,4}，`key=['M','N','K']` | 先用本表剪枝到 ≤20 个 config，全网格爆搜浪费编译时间 |

**Triton 专用排查**：
```bash
# 看 Triton 编译产物（确认 cp.async/TMA/mma 是否真的生成）
python -c "...; print(kernel.asm['ptx'])"        # 或 .asm['ttgir'] / .asm['sass']
# 确认寄存器/spill：
print(kernel.n_regs, kernel.n_spills)             # n_spills > 0 → P4
# torch.compile 生成的 kernel：TORCH_LOGS=output_code 导出后单独 benchmark
```

<!-- v2 refine 2026-06-10：新增规则 9（组合 pattern 库）+ 规则 10（Triton 映射表） -->
