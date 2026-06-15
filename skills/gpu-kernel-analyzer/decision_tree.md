# 诊断决策树

> gpu-kernel-analyzer 的完整分析逻辑，按层次递进

---

## 入口：数据类型路由

```
用户有数据？
├── ncu 数据 → 【NCU 分析树】（下方）
├── nsys 数据 → 【NSYS 分析树】（第三节）
├── 两者都有 → nsys 先找热点 kernel → ncu 深挖
└── 只有症状 → 【症状问诊表】（第四节）
```

---

## 【NCU 分析树】

### 节点 0：数据质量预检

```
kernel 执行时间 < 20µs？
  → YES: 提示数据可能不稳定，建议增大 batch size 再 profile
  → NO:  继续

有任何指标超过 100% 或为负？
  → YES: 可能是 replay 误差，提示用户（但继续分析）
  → NO:  继续
```

---

### 节点 0.1：NCU Rule Engine（先于一切手动分析）

```bash
# 如果用户有 .ncu-rep 文件，立刻导出 details 页面
ncu --import profile.ncu-rep --page details > details.txt
```

阅读 details.txt，找到所有 `OPT   Est. Speedup: X%` 条目，**按 X 降序排列**。  
这些规则引擎建议往往直接指向根因，不要在看完规则引擎之前手动猜测。

若规则引擎的最高建议 speedup > 30%：直接优先处理该条，跳过后续节点。  
若规则引擎建议均 < 10%：继续常规分析流程。

---

### 节点 0.3：Grid 大小预检（SOL 分析前必查，30 秒完成）

```
launch__grid_size 是否 < GPU SM 数？
  A100=108, H100=132, RTX 4090=128

YES（Grid 太小）:
  → 诊断：SMALL GRID（部分 SM 全程空闲）
  → Est. Speedup 往往 50-90%
  → 修复方向：增大 Grid（Split-K / Persistent Kernel / 减小 tile size）
  → ⚠️ 先解决这个再做其他分析，其他 SOL 数据在 small grid 下会失真

NO → 继续

额外检查：launch__waves_per_multiprocessor < 1.5 时关注末尾波浪浪费
（例：1.1 waves → 最后一波只用了 10% SM → tail 浪费 ~45%）
```

---

### 节点 0.5：Roofline 上界预计算（建议优化开始时执行一次）

**目的**：在进入 SOL 分析前，先确定理论极限和当前效率，避免在已接近极限的 kernel 上过度投入。

**采集命令**：
```bash
ncu --section SpeedOfLight_RooflineChart \
    --section SpeedOfLight \
    --csv --log-file roofline.csv ./kernel

# 或手动采集（KernelAgent 验证指标集，单 pass，精确）
ncu --metrics \
  sm__throughput.avg.pct_of_peak_sustained_elapsed,\
  gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,\
  sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,\
  dram__bytes_read.sum,dram__bytes_write.sum,\
  gpu__time_duration.sum \
  --csv ./kernel
```

**计算逻辑（KernelAgent SOL 方法，无需手算 FLOP）**：
```python
# 主效率指标：直接用 SOL 百分比
compute_sol = sm__throughput_pct              # sm__throughput.avg.pct_of_peak_sustained_elapsed
memory_sol  = gpu__compute_memory_pct         # gpu__compute_memory_throughput.avg.pct... (L2+DRAM 综合)
efficiency  = max(compute_sol, memory_sol)    # 主效率，0-100

uses_tc = sm__pipe_tensor_cycles_active_pct > 5.0

# 瓶颈三分类
if memory_sol < 60 and compute_sol < 60:  bottleneck = "underutilized"
elif memory_sol >= compute_sol:            bottleneck = "memory"
else:                                      bottleneck = "compute"

# 可选：AI-based 方法（用于 Roofline 图可视化，需 FLOP 计数）
# total_flop = 2*M*N*K  或  ffma*2 + fadd + fmul + hfma*2 + hmma*8192
# AI = total_flop / (dram_read + dram_write)
```

> **关键**：`memory_sol` 用 `gpu__compute_memory_throughput`（L2+DRAM 综合指标），不是 `dram__throughput`（仅 HBM）。这是 KernelAgent `ncu_roofline.py` 的验证标准。

**硬件参考**（供 AI-based 方法参考）：

| GPU | FP32 Ridge (F/B) | TC Ridge (F/B) | 显存 BW |
|-----|-----------------|----------------|---------|
| A100 SXM4 | 9.75 | 156 | 2.0 TB/s |
| H100 SXM5 | 20 | 295 | 3.35 TB/s |
| RTX 4090 | 82 | 165 | 1.0 TB/s |

**决策**：
```
efficiency >= 95%  → at_roofline = True，输出"已达 Roofline 极限，停止优化"
efficiency < 95%   → 记录 efficiency，继续进入 SOL 分类
  收敛检查：连续 5 轮改善 < 0.1% → 也停止（即使未达 95%）
efficiency < 50%   → 有明显优化空间，优先解决当前瓶颈
```

---

### 节点 1：SOL 分类（强制首步）

> **指标区分**：
> - `dram__throughput`（HBM 带宽 SOL）→ 用于此处 DRAM 层诊断分类
> - `gpu__compute_memory_throughput`（L2+DRAM 综合 SOL）→ 用于节点 0.5 Roofline 效率
> 两者不要互换。Memory-bound kernel 的 `gpu__compute_memory_throughput` 通常高于 `dram__throughput`（因含 L2 流量）。

读取（从 ncu Details Page → GPU Speed Of Light）：
- `sm_pct` = sm__throughput %
- `l1_pct` = l1tex__throughput %  
- `l2_pct` = lts__throughput %
- `dram_pct` = dram__throughput %   ← HBM 层诊断用此指标

```
分类逻辑：

sm_pct > 60 AND dram_pct < 50
  → COMPUTE-BOUND → 节点 2A

dram_pct > 70
  → MEMORY-BOUND (DRAM) → 节点 2B-DRAM

l2_pct > 85 AND dram_pct ≤ 70
  → MEMORY-BOUND (L2) → 节点 2B-L2

l1_pct > 85 AND l2_pct ≤ 70
  → MEMORY-BOUND (L1/Shared) → 节点 2B-L1

全部 < 40
  → LATENCY-BOUND → 节点 2C

其他（多个 >50 但没有主导）
  → MIXED → 依次执行 2B 再 2A，取影响更大者
```

**立即输出**：`> 诊断：[类型]，主要限制单元：[单元]`

---

### 节点 1.5：PM 采样时间线 + Tail Effect 独立检查

**采集命令（所有 bottleneck 类型均适用）**：
```bash
ncu --set full --section PmSampling --section PmSampling_WarpStates \
    -k "regex:KERNEL_NAME" -c 1 -o profile.ncu-rep ./kernel
# 在 Nsight Compute GUI 中查看 PM Sampling 标签页
```

**四种时间线形态 → 独立于 SOL 分类，任何 bottleneck 都可能出现**：

| 形态 | 视觉特征 | 含义 | 处理 |
|------|---------|------|------|
| **平顶陡降** | 高位平稳 → 骤降至 0 | ✅ 理想，Grid 充足均衡 | 无 |
| **平顶渐降** | 高位 → 末尾逐渐下滑 | ⚠️ 可变长度 Tail Effect | Sorted Batching / Split-K |
| **持续低位** | 全程低 SM 利用率 | ❌ Grid 太小 / 严重 stall | 增大 Grid（见节点 0.3）|
| **锯齿波动** | 计算↑ 内存↑ 交替 | ⚠️ 无 compute-memory overlap | Double buffering / async copy |

**可变长度 Tail Effect（高价值优化，经常被忽视）**：
```
触发条件：平顶渐降形态 + 输入含不同长度序列

验证指标：
  max_seq_len / avg_seq_len > 3x → 高风险
  SM active cycles: max/min > 5x → 确认不均衡

根因：各 CTA 的内层循环次数由序列长度决定
       最长序列 CTA 决定整体延迟，其余 SM 提前空闲

修复优先级（同 OPT-TAIL-01）：
  1. Sorted Batching — 同 wave 内序列长度相近
  2. Split-K — 长序列拆分到多个 CTA，最后 reduce
  3. Persistent Kernel — work-stealing 动态分配
```

> ⚠️ Tail Effect 是**独立于 SOL 分类**的问题。COMPUTE-BOUND / MEMORY-BOUND kernel 同样可能有 tail effect，不仅仅是 LATENCY-BOUND。

---

### 节点 2A：COMPUTE-BOUND 深分析

**Step 0：TC 未使用检查（Pattern F，先于 Pipeline 检查）**
```
sm_pct > 60% AND smsp__pipe_tensor_cycles_active < 5%？
  → YES + workload 是矩阵乘法形状（M×K×N）?
    → 警告：Tensor Core 完全未使用！FP32 FMA 在跑，但 TC 吞吐是 FP32 的 16-32x
    → 优先处理：改用 wmma / mma PTX / cuBLAS，预期加速 5-20x
    → 检查：数据类型是否为 FP32？改 FP16/BF16 才能走 TC
    → 见 optimization_db.md OPT-COMP-TC
  → NO → 继续 Step 1
```

**Step 1：Pipeline 利用率**  
读取（Compute Workload Analysis section）：

```
smsp__pipe_fma_cycles_active > 80%
  → FP32 FMA 饱和 → 节点 3A-FMA

smsp__pipe_tensor_cycles_active > 80%
  → Tensor Core 饱和 → 节点 3A-TC（好事！说明 TC 利用好）

smsp__pipe_fp64_cycles_active > 80%
  → FP64 饱和 → 节点 3A-FP64（警告：double 代价高）

smsp__pipe_l1tex_cycles_active > 80% 且 sm_pct 高
  → LSU 内存指令密集 → kernel 是内存指令主导的 compute → 节点 3A-LSU

全部 < 60%
  → IPC 低，非 pipeline 饱和 → 节点 3A-IPC
```

**Step 2：IPC 检查**  
`smsp__inst_executed.avg.per_cycle_active`
- IPC > 3 → 发射率好，瓶颈是算法级（已接近峰值）
- IPC 1-2 → 正常
- IPC < 1 → 严重低效 → 检查 branch divergence

**Step 3：Branch Divergence**  
`smsp__thread_inst_executed_per_inst_executed.ratio`
- < 0.8 → 显著 divergence → 节点 3A-DIV

---

### 节点 2B-DRAM：MEMORY-BOUND (DRAM 带宽) 深分析

**Step 1：理论 vs 实际 DRAM 流量**
```
理论 = 输入 tensor bytes + 输出 tensor bytes
实际 = dram__bytes_read.sum + dram__bytes_write.sum

ratio = 实际 / 理论
ratio < 1.2 → 数据基本无重复加载，带宽利用合理，减少数据量是方向
ratio 1.2-3 → 有重复加载，改善缓存复用
ratio > 3   → 严重缓存不友好，需要 tiling 重构
```

**Step 2：L2 命中率**  
`lts__t_sector_hit_rate.pct`
- < 30% → L2 miss 严重 → 数据不适合 L2 大小，需要 blocking/tiling
- 30-60% → 中等，考虑增大 tile size
- > 60% → L2 ok，问题是 working set 本身太大

**Step 3：是否有 coalescing 问题导致额外 DRAM 流量**  
`l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio`
- > 4 → 访问不连续，DRAM 流量被放大 → 改 SoA 布局，对齐访问

---

### 节点 2B-L2：MEMORY-BOUND (L2 带宽) 深分析

主要原因：L1 miss rate 高，大量请求打到 L2

**Step 1：L1 命中率**  
`l1tex__t_sector_hit_rate.pct`
- < 50% → L1 不够用 → 用 Shared Memory 做手动 tiling
- 50-80% → 中等，优化数据局部性

**Step 2：Bank Conflict（可能导致 L1 压力）**  
`l1tex__data_bank_conflicts_pipe_lsu.sum`
- 与 shared load 请求数的比率 > 10% → padding 修复

---

### 节点 2B-L1：MEMORY-BOUND (L1/Shared) 深分析

```
l1tex__data_bank_conflicts_pipe_lsu 高？
  → YES: Bank Conflict → Padding 方案（见 optimization_db.md）
  → NO:  Shared Memory 带宽本身达到极限（少见）
         → 减少 smem 操作频率，考虑 register 替代
```

---

### 节点 2C：LATENCY-BOUND 深分析

**Step 1：Occupancy 量化**
```
achieved = sm__warps_active.avg.pct_of_peak_sustained_active
theoretical = sm__maximum_warps_per_active_cycle_pct
gap = theoretical - achieved

gap > 20%  → 运行时不均衡（某些 SM 负载不平）
gap < 5%   → 受静态资源限制 → Step 2
```

**Step 2：Occupancy Limiter**  
查看哪个 `launch__occupancy_limit_*` 最小（即哪个资源最先耗尽）：

```
registers  → 每线程寄存器过多 → __launch_bounds__
shared_mem → 每块 smem 过多  → 减少 smem 或动态分配  
warps      → block size 太小 → 至少 128 threads/block
blocks     → blocks/SM 限制  → 通常无需干预
barriers   → 命名 barrier 数量 → 减少 __bar.sync / mbarrier 使用数
```

> `launch__occupancy_limit_barriers`：当 kernel 使用 PTX named barriers（`barrier.sync` / `cp.async.mbarrier`）时，barrier 数量本身也会成为 occupancy limiter，Hopper+ 尤其需要关注。

**Step 3：Warp Stall 根因**  
查看 Warp State Statistics，找占比 >15% 的 stall：

```
long_scoreboard (占比 >20%)
  → Global memory latency 主导
  → 方案：增大 occupancy + shared memory tiling + prefetch

short_scoreboard (占比 >15%)
  → L1/Const cache latency
  → 方案：重排指令，增加指令间距

math_throttle
  → Compute pipeline 满（正常！升级到 2A 处理）

wait (占比 >15%)
  → Barrier 同步开销大
  → 方案：减少 __syncthreads；改 __syncwarp

mio_throttle / lg_throttle (占比 >15%)
  → Memory IO 队列满
  → 方案：合并 load/store；减少内存指令密度

no_instruction (占比 >10%)
  → Occupancy 不够，没有 warp 可发射
  → 强制增大 occupancy（最重要）

membar (占比 >10%)
  → 过多 __threadfence()
  → 用 relaxed atomic；减少 fence 操作
```

**Step 4：Waves 浪费检查**  
`launch__waves_per_multiprocessor`
- 小数部分 < 0.5 → 最后一波严重浪费 → 调整 grid size

> **可变长度 Tail Effect**：已移至独立的**节点 1.5**（适用于所有 bottleneck 类型）。  
> 此处仅检查固定 grid 的 wave 浪费，可变长度问题见节点 1.5。

---

### 节点 2T：Triton Kernel 专用路由（kernel 是 Triton 写的 → 诊断后走此节点）

> Triton 的优化自由度被压缩为 meta-parameters，诊断结论按下表直接映射为参数行动。
> 完整映射表见 gpu-kernel-diag skill `diag_rules.md` 规则 10。核心对应：

```
诊断结论                    → Triton 行动
─────────────────────────────────────────────────────────────
long_scoreboard 高（2C）    → num_stages 2→3→4（自动生成 cp.async/TMA 流水线）
limiter=shared_mem（2C）    → num_stages↓ 或 BLOCK_K↓（smem ≈ (BM+BN)×BK×dtype×stages）
limiter=registers（2C）     → num_warps↓（8→4）或 BLOCK↓；kernel.n_spills>0 必须先处理
grid 太小（节点 0.3）       → BLOCK_M/N↓ 或 Split-K；grid ≥ 2×SM 数
sectors/req 高（2B）        → 连续维度必须由 tl.arange 最后一维索引；输入 .contiguous()
TC 未使用（2A Step 0）      → 用 tl.dot（勿手写乘加）；fp16/bf16；BLOCK ≥16 且 16 倍数
TC 等数据（2A+2B 混合）     → BLOCK_K↑ + num_stages≥3
不确定                      → @triton.autotune，先用上表剪枝到 ≤20 个 config
─────────────────────────────────────────────────────────────
⚠️ 一次只动一个参数（BLOCK/num_stages/num_warps 对 smem 和寄存器是相乘压力）
验证手段：print(kernel.n_regs, kernel.n_spills)；kernel.asm['ptx'] 确认 cp.async/wgmma 生成
```

**架构分支**（详表见 metrics_reference.md 架构差异表）：
```
Hopper(sm_90)+ : TMA/wgmma 由 Triton 自动启用；手写 CUDA 才需要 cuda::memcpy_async + mbarrier
Ampere(sm_80)  : cp.async 可用，无 TMA；建议给出 cp.async 版本代码
更老/消费级    : 仅同步拷贝 + double buffering；smem 预算按 100KB 算（AD102）
```

---

## 节点 3A 系列：Compute 优化具体路径

### 3A-FMA：FP32 FMA 饱和
这通常是好事（compute 充分利用）。只有算法层面可以提升：
- 减少计算量（算法改进）
- Kernel fusion（减少 launch overhead）
- 检查是否有冗余计算

### 3A-TC：Tensor Core 饱和
✅ 好事，TC 被充分使用。优化方向：
- 确认 GEMM 形状满足 TC tile 对齐（M、N、K 是 16 的倍数）
- 检查数据类型是 FP16/BF16 而非 FP32（Volta/Turing）
- 减少 TC 之外的 FP32 开销（避免频繁 format conversion）

### 3A-FP64：FP64 饱和
⚠️ 通常是问题。检查：
- 是否有 double 常量未标 `f`（如 `0.1` vs `0.1f`）
- 是否真的需要 double 精度？改 float 可获 2-32x 加速

### 3A-DIV：Branch Divergence
```
优化策略：
1. Warp 内线程统一分支
   - 按 warp 大小（32）对齐数据，让每个 warp 只走一条路径
2. 用预测代替分支
   - 用 ternary / 位运算代替 if-else
   - `val = cond ? a : b;` → 编译器可能生成 SELP 指令（无 divergence）
3. 数据重排
   - Sort/partition 数据使相同分支的元素连续
```

---

## 【NSYS 分析树】

### NS 节点 0：Expert System 自动规则（先于一切手动分析）

```bash
# 第一步：采集
nsys profile --stats=true -o report ./your_app

# 第二步：Expert System（等价于 NCU 的 Rule Engine）
nsys analyze report.nsys-rep
```

Expert System 当前有 **6 条内置规则**，自动检测：

| 规则 | 问题描述 | 建议修复 |
|------|---------|---------|
| `cuda_memcpy_async` | Async memcpy 但内存是 pageable → 退化为同步 | 改用 pinned memory |
| `cuda_memcpy_sync` | 同步 memcpy 阻塞 host | 改用 `cudaMemcpyAsync` |
| `cuda_memset_sync` | 同步 memset 阻塞 host | 改用 `cudaMemsetAsync` |
| `cuda_api_sync` | `cudaDeviceSynchronize` 等阻塞 host | 改用 event-based sync |
| `gpu_gaps` | GPU idle > 500ms | 找 CPU 侧阻塞原因 |
| `gpu_time_util` | GPU 利用率低（分时间段检测） | 减少 CPU-GPU gap |

> 与 NCU 的 Rule Engine 一样：先看 Expert System，再手动分析。

---

### NS 节点 0.5：PM 采样时间线

> **注意**：PM 采样是 **ncu** 工具（不是 nsys），完整分析见【NCU 分析树】→ 节点 1.5。
> 此处仅提醒：从 nsys 发现 kernel 热点后，应切换到 ncu + PmSampling 做时间线分析。

---

### NS 节点 1：热点 Kernel 定位

**用 `nsys stats` 快速找热点**（命令行，无需打开 GUI）：

```bash
# 方式 1：直接采集时输出统计
nsys profile --stats=true -o report ./your_app

# 方式 2：事后统计（已有 .nsys-rep 时）
nsys stats report.nsys-rep --report cuda_gpu_kern_sum    # 按 Total Time 排序的 kernel 热点表
nsys stats report.nsys-rep --report cuda_gpu_mem_time_sum # 内存操作耗时
nsys stats report.nsys-rep --report cuda_api_gpu_sum      # CUDA API + kernel 总览
nsys stats report.nsys-rep --report osrt_sum              # OS runtime (pthreads, etc.)
```

`cuda_gpu_kern_sum` 输出字段含义：
```
Time%     - 该 kernel 占所有 kernel 总时间的百分比
Total Time - 所有实例的总执行时间
Instances  - 调用次数
Avg/Med    - 平均/中位执行时间
Min/Max    - 最快/最慢实例
StdDev     - 时间标准差（高 → 执行时间不稳定）
Name       - Kernel 名称
```

**决策**：
```
找 Time% 最高的 kernel → 用 ncu 深挖
Time% > 50% 且 Instances 多 → 优先优化
StdDev 异常高 → 考虑 cuda_gpu_kern_pace 分析（见 NS 节点高级）
```

### NS 节点 2：GPU 利用率分析

```
查看 Timeline 或 gpu_time_util rule 输出：
GPU active time / total time = GPU 利用率

GPU 利用率 > 80% → 查热点 kernel（NS 节点 1）→ 用 ncu 深挖
GPU 利用率 < 80% → 有 CPU-GPU 交互瓶颈 → NS 节点 3
```

**Queue Time 分析（`cuda_kern_exec_sum`）**：
```bash
nsys stats report.nsys-rep --report cuda_kern_exec_sum
```
- `QAvg`（Queue Time 平均）= API 返回到 kernel 开始执行的延迟
- Queue Time 高 → GPU 繁忙排队，launch overhead 大，或 CPU 发射太慢
- Queue Time 接近 0 → GPU 空闲等 CPU（CPU 是瓶颈）

### NS 节点 3：CPU-GPU 交互分析

```
GPU idle gap 的原因（Expert System 会自动标记大部分）：
├── pageable memcpy → 见 Expert System cuda_memcpy_async 规则
├── 同步 cudaMemcpy/Memset → 见 Expert System 规则
├── cudaDeviceSynchronize 过多 → 见 Expert System cuda_api_sync 规则
├── cudaMalloc/Free 频繁调用 → 用内存池
└── CPU 计算阻塞 GPU launch → 生产者/消费者流水线
```

### NS 节点 4：多 Stream 分析

```
多个 stream 但 kernel 顺序执行？
└── 检查 stream 间是否有 cudaEvent 强制串行化
└── 检查 cudaStreamSynchronize 调用位置

存在 kernel 间明显 gap？
└── 检查两个 kernel 之间的 CPU 代码是否耗时
└── 考虑 CUDA Graph 消除 CPU launch overhead
```

### NS 节点高级：Recipe 分析

```bash
# 多文件对比（优化前后）
nsys recipe diff --input run1.nsys-rep run2.nsys-rep --output diff_dir

# Kernel 执行时间稳定性分析
nsys recipe cuda_gpu_kern_pace --input report.nsys-rep --output pace_dir

# GPU 利用率热力图
nsys recipe cuda_gpu_time_util_map --input report.nsys-rep --output heatmap_dir

# GPU idle gap 详细分析
nsys recipe gpu_gaps --input report.nsys-rep --output gaps_dir

# NCCL 通信 vs 计算 overlap
nsys recipe nccl_gpu_overlap_trace --input report.nsys-rep --output nccl_dir
```

---

## 【症状问诊表】

| 用户描述 | 最可能根因 | 推荐采集命令 | 确认指标 |
|---------|----------|------------|---------|
| "GPU 利用率只有 30%" | CPU 瓶颈或数据传输 | `nsys profile` | Timeline 中 GPU active % |
| "Tensor Core 利用率低" | GEMM 形状不对齐 | `ncu --set full` | `smsp__pipe_tensor_cycles_active` |
| "比 cublas 慢 3x" | 内存访问模式差 | `ncu --section MemoryWorkloadAnalysis` | L1/L2 hit rate, sectors/request |
| "训练 GPU 利用率 60%" | DataLoader CPU 瓶颈 | `nsys profile` | CPU timeline 中 Python 时间 |
| "Occupancy 很低" | 寄存器或 smem 过多 | `ncu --set basic` | `launch__occupancy_limit_*` |
| "kernel 时间差异大" | Branch divergence | `ncu --section SourceCounters` | Thread efficiency ratio |
