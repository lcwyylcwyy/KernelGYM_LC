# Nsight Compute 指标字典（学习版）

> 每个指标含：硬件含义、读取路径、正常范围、异常含义、优化方向

---

## 第一层：Speed of Light（SOL）— 必须最先看

### `sm__throughput.avg.pct_of_peak_sustained_elapsed`
**Section**：GPU Speed Of Light  
**读取位置**：Details Page → GPU Speed Of Light → Compute  
**含义**：SM（流多处理器）计算吞吐量占理论峰值的百分比  
**计算**：实际每周期发射的指令数 / SM 能发射的理论最大指令数 × 100%  
**正常范围**：
- >80% = Compute-Bound（计算密集，正常现象）
- 40-80% = 部分利用
- <40% = 严重未充分利用  

**类比**：工厂机器的开动率。99% 说明机器全力运转。但注意——如果工厂等待原材料（内存数据），机器表面在"工作"但实际上在空转等待。

**关联**：必须与 `dram__throughput` 联合看。若 SM 高但 DRAM 也高，说明 kernel 真的计算密集；若只有 SM 高而 DRAM 低，可能是 latency 被误计入。

---

### `l1tex__throughput.avg.pct_of_peak_sustained_elapsed`
**Section**：GPU Speed Of Light  
**含义**：L1 Cache / Texture Cache / Shared Memory 复合吞吐量利用率  
**正常范围**：>85% = L1 层级是瓶颈  
**含义**：Shared Memory bank conflict 或 大量 texture fetch 导致 L1 单元繁忙  

---

### `lts__throughput.avg.pct_of_peak_sustained_elapsed`
**Section**：GPU Speed Of Light  
**含义**：L2 Cache（LTS = Level Two Slice）吞吐量利用率  
**正常范围**：>85% = L2 是瓶颈  
**含义**：L1 miss rate 高，大量请求穿透到 L2；或 L2 working set 超过容量  

---

### `dram__throughput.avg.pct_of_peak_sustained_elapsed`
**Section**：GPU Speed Of Light  
**含义**：设备内存（HBM/GDDR）读写带宽利用率  
**正常范围**：>70% = DRAM 带宽饱和（Memory-Bound）  
**含义**：数据无法在缓存中复用，每次都要去 DRAM 取  

---

## 第二层：Compute Workload Analysis — Compute-Bound 时看

### `smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed`
**Section**：Compute Workload Analysis  
**含义**：FP32 FMA（Fused Multiply-Add）流水线活跃周期比  
**物理单元**：FMA Heavy + FMA Lite 管道（处理 FADD/FMUL/FFMA 指令）  
**正常范围**：>80% = FP32 计算饱和  
**关联**：这是通用 CUDA kernel 最常见的饱和点

---

### `smsp__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed`
**Section**：Compute Workload Analysis  
**含义**：Tensor Core 管道活跃周期比  
**物理单元**：Tensor Core（专为矩阵乘法设计，处理 WMMA/MMA 指令）  
**正常范围**：GEMM 优化好的 kernel 应 >80%  
**低的原因**：
- GEMM 矩阵形状不对齐到 Tensor Core tile（M/N/K 需是 16 的倍数）
- 数据类型非 FP16/BF16/INT8（TC 不处理 FP32 GEMM，Ampere 起例外）
- 算法不使用 WMMA/CUTLASS API

---

### `smsp__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_elapsed`
**含义**：FP64（double）流水线利用率  
**注意**：FP64 单元数量远少于 FP32（A100 FP64:FP32 = 1:2，消费级 GPU 更少）  
**高的原因**：代码中有 double 类型运算，考虑改 float

---

### `smsp__pipe_l1tex_cycles_active.avg.pct_of_peak_sustained_elapsed`
**含义**：LSU（Load/Store Unit）管道活跃周期比，反映内存指令密度  
**注意**：这个高但 `sm__throughput` 也高，说明 kernel 在 compute 和 mem 指令间交替 → 可能是混合型

---

### `smsp__inst_executed.avg.per_cycle_active`（IPC）
**含义**：每个 SM 活跃周期平均执行的指令数（Instruction Per Cycle）  
**计算**：总指令数 / 活跃周期数  
**正常范围**：Ampere SMSP 理论最大 ~4（每 SMSP 每周期发射 1 条 × 4 SMSP）  
- >3 = 非常好的指令吞吐
- 1-2 = 正常
- <1 = 指令层面效率低（频繁 stall）

---

## 第三层：Memory Workload Analysis — Memory-Bound 时看

### `l1tex__t_sector_hit_rate.pct`
**Section**：Memory Workload Analysis  
**含义**：L1 Cache 扇区命中率  
**计算**：L1 命中的扇区数 / L1 总请求扇区数 × 100%  
**正常范围**：
- >80% = 数据很好地在 L1 复用
- 50-80% = 有改善空间
- <50% = L1 完全失效，数据每次都从 L2 取  

**优化**：用 Shared Memory 手动缓存热点数据，减少重复的 L1 miss

---

### `l1tex__data_bank_conflicts_pipe_lsu.sum`
**含义**：Shared Memory 访问产生的 bank conflict 次数  
**物理背景**：Shared Memory 有 32 个 bank，同一个 warp 内多个线程访问同一 bank 的不同地址 → 串行化  
**判断**：此值 / shared memory 请求总数 > 10% → 显著 conflict  
**优化**：在 shared memory 数组末尾 padding 1 个元素（`float smem[ROW][COL+1]`）

---

### `l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio`
**含义**：每次 Global Memory Load 请求平均访问的 L1 扇区数  
**物理背景**：理想情况下 warp 内 32 线程访问连续 128B → 1 request, 4 sectors（=1 cache line）  
**正常范围**：
- ~4 = 完美 coalesced（128B 对齐）
- ~32 = 完全非合并（每个线程访问独立 cache line）
- 通常看这个比值：好 ~1 sector/request（在 sector 级别），坏 ~8+

**优化**：将数据结构从 AoS（Array of Structures）改为 SoA（Structure of Arrays）

---

### `lts__t_sector_hit_rate.pct`
**含义**：L2 Cache 扇区命中率  
**正常范围**：
- >60% = L2 有效缓存
- <30% = Working set 超过 L2 容量，数据直通 DRAM  

**注意**：L2 miss 的直接后果是 DRAM 带宽压力增加

---

### `dram__bytes_read.sum` / `dram__bytes_write.sum`
**含义**：kernel 执行期间读写 DRAM 的实际字节数  
**用途**：与理论 working set 对比，判断数据复用效果  
**计算理论 working set**：输入 tensor 大小 + 输出 tensor 大小  
**超额比 > 2x** → 存在严重的重复加载，缓存利用差

---

## 第四层：Occupancy — Latency-Bound 时看

### `sm__warps_active.avg.pct_of_peak_sustained_active`（Achieved Occupancy）
**Section**：Occupancy  
**含义**：实际活跃 warp 数 / SM 能容纳的最大 warp 数  
**正常范围**：
- >75% = 好
- 50-75% = 可接受
- <25% = 严重不足，延迟隐藏能力极差  

**常见误区**：Occupancy 高不代表性能好，但 Occupancy 低一定会导致性能差。

---

### `sm__maximum_warps_per_active_cycle_pct`（Theoretical Occupancy）
**含义**：基于 kernel 静态参数（寄存器/smem/block size）计算的理论最大 occupancy  
**用途**：与 Achieved Occupancy 对比，差距大 → 动态调度问题；差距小 → 受静态资源限制

---

### `launch__occupancy_limit_registers`
**含义**：由于每线程寄存器数量过多导致的 occupancy 上限  
**判断**：若此值是限制因素，意味着 `launch__registers_per_thread` 很高  
**优化**：用 `__launch_bounds__(max_threads, min_blocks)` 提示编译器减少寄存器使用

---

### `launch__occupancy_limit_shared_mem`
**含义**：由于每 block shared memory 用量过大导致的 occupancy 上限  
**优化**：减少静态 shared memory 大小；改用动态分配并在运行时控制量

---

### `launch__waves_per_multiprocessor`
**含义**：整个 grid 需要执行多少轮（wave）才能完成  
**计算**：ceil(grid_size / (SM 数量 × blocks/SM))  
**重要性**：最后一轮（wave）如果 block 很少 → 大量 SM 空转（tail effect）  
**优化**：调整 grid size 让最后一轮尽量饱满

---

## 第五层：Scheduler Statistics — 调度效率分析

### `smsp__warps_eligible.avg.per_cycle_active`
**含义**：每周期平均有多少 warp 处于 eligible（就绪可发射）状态  
**正常范围**：越高越好；若 < 1 → 调度器经常没有 warp 可发射

---

### `smsp__issue_active.avg.per_cycle_active`
**含义**：每周期调度器实际发射指令的比率  
**理想值**：接近 1.0（每周期都发射）  
**低的含义**：大量 issue slots 被浪费（没有 eligible warp）

---

## 第六层：Warp State Statistics — Stall 原因分析

查看 Details Page → Warp State Statistics 各 stall 原因的占比（cycles 分布）。

| Stall 类型 | 物理含义 | 根因 | 优化 |
|-----------|---------|------|------|
| `long_scoreboard` | 等待 L2/DRAM 数据（长延迟，300-800 cycles）| Global memory latency | 增大 occupancy；shared memory tiling；prefetch |
| `short_scoreboard` | 等待 L1/Const Cache（中延迟，20-100 cycles）| L1 miss 或 constant cache miss | 重用数据；调整访问模式 |
| `math_throttle` | 数学流水线全满 | ✅ Compute-Bound 的正常状态 | 不需要优化 |
| `wait` | __syncthreads barrier | 同步点不均衡（某些线程先到，等其他线程）| 减少 sync；重构算法 |
| `mio_throttle` | Memory IO 队列满 | 内存指令过于密集 | 合并 load/store；减少指令数 |
| `lg_throttle` | Local/Global 内存队列满 | 大量 global 或 local memory 操作 | 改用 shared memory；减少访问次数 |
| `no_instruction` | 调度器没有任何 warp 可以发射 | Occupancy 极低，warp 全部 stalled 或不存在 | 增大 block size；减少资源占用 |
| `not_selected` | Warp eligible 但未被选中 | 竞争激烈（正常，有多个 eligible warp）| 无需优化，这是健康现象 |
| `membar` | 等待 memory fence 完成 | `__threadfence()` 过多 | 用 relaxed semantic atomic；减少 fence |
| `drain` | Warp 完成工作后退出 | Grid 末尾的 tail effect | 调整 grid size |

---

## 第七层：Source Counters — 源码级分析

### `smsp__thread_inst_executed_per_inst_executed.ratio`（Thread Efficiency）
**含义**：每条发射的指令中，平均有多少线程真正在执行（非 predicated off）  
**理想值**：1.0（32/32 线程都执行）  
**低的含义**：Branch Divergence——warp 内线程走了不同分支，只有部分线程实际执行

---

## 快速查阅表

| 分析层次 | 首选指标 | 异常阈值 | 下一步 |
|---------|---------|---------|-------|
| SOL 分类 | `sm__throughput`, `dram__throughput` | SM>60% → Compute；DRAM>70% → Memory | 对应路径深入 |
| Compute | `smsp__pipe_*_cycles_active` | >80% = 饱和 | 确认哪条 pipeline |
| Memory L1 | `l1tex__t_sector_hit_rate` | <50% = 差 | 考虑 shared memory |
| Memory L1 | `l1tex__data_bank_conflicts` | >10% = 明显 | Padding smem |
| Memory L2 | `lts__t_sector_hit_rate` | <30% = 差 | 数据 tiling |
| Occupancy | `sm__warps_active.*pct` | <50% = 低 | 查 limiters |
| Stall | Warp State 分布 | >30% 某原因 | 按 stall 类型优化 |
