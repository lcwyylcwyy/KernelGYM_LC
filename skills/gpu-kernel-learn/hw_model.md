# GPU 硬件模型参考

> 讲解 GPU 性能指标时的硬件基础知识

---

## 1. 执行层次结构

```
Grid
 └─ Block (CTA, Cooperative Thread Array)
     └─ Warp (32 threads, 最小调度单位)
         └─ Thread
```

**关键规则**：
- 一个 CTA 内所有 warp 必须在**同一个 SM** 上运行
- Warp 是调度的原子单位——32 个线程同时执行同一条指令（SIMT）
- CTA 之间相互独立，无法跨 CTA 同步（只能通过全局内存 + atomic）

---

## 2. SM（Streaming Multiprocessor）结构

```
SM
├── 4× SMSP（SM Sub-Partition，主要处理单元）
│   ├── Warp Scheduler（每个 SMSP 1个，每周期可发射 1-2 条指令）
│   ├── Register File（32-bit 寄存器，Ampere: 每 SMSP 16K 寄存器）
│   └── 执行单元（每个 SMSP 各有）
│       ├── FP32 FMA 单元（FADD/FMUL/FFMA）
│       ├── INT32 ALU 单元
│       ├── FP64 单元（数量远少于 FP32，A100: FP64:FP32 = 1:2）
│       ├── Tensor Core（MMA 指令，矩阵运算）
│       ├── LSU（Load/Store Unit，内存指令）
│       └── SFU（Special Function Unit，sin/cos/rcp）
└── 共享（所有 SMSP 共用）
    ├── L1 Cache / Shared Memory（同一块物理 SRAM，可配比例）
    ├── Texture Unit
    └── RT Core（光追专用，非 CUDA 通用）
```

**SMSP 的 Warp Pool**：
- Volta/Turing：每 SMSP 最多 16 个活跃 warp
- Ampere+：每 SMSP 最多 16 个活跃 warp
- 每周期调度器从 pool 中选 1 个 eligible warp 发射指令
- Warp 只有在所有输入就绪时才是 eligible（否则 stalled）

---

## 3. Warp 生命周期与 Stall

```
Warp 状态机：
  Active → Eligible → Selected → Issued
              ↕
           Stalled（等待数据/计算/同步）
```

**Latency Hiding 原理**：SM 同时驻留多个 warp，当一个 warp stall 时，调度器切换到另一个 eligible warp。warp 越多，latency 越容易被隐藏。

这就是为什么 **Occupancy（活跃 warp 比例）**很重要——warp 不够时无法隐藏延迟。

---

## 4. 内存层次结构（从快到慢）

```
寄存器 (~1 cycle)
    ↓
Shared Memory / L1 Cache (~20-30 cycles)    ← 片上，每 SM 独享
    ↓
L2 Cache (~200 cycles)                       ← 全局共享，所有 SM 共用
    ↓
DRAM / HBM (~600-800 cycles)                 ← 显存，最慢但最大
    ↓
PCIe / NVLink → CPU Memory (~microseconds)
```

**各层容量（A100 为例）**：
| 层级 | 延迟 | 带宽 | 容量 |
|------|------|------|------|
| L1 + Shared | ~20 cy | ~19 TB/s (per SM) | 192 KB/SM |
| L2 | ~200 cy | ~4 TB/s | 40 MB |
| HBM2e | ~800 cy | ~2 TB/s | 80 GB |

---

## 5. 内存类型

| 类型 | 位置 | 访问范围 | 特点 |
|------|------|----------|------|
| **Registers** | SM 片上 | 线程私有 | 最快；溢出 → local memory（慢！）|
| **Shared Memory** | SM 片上 | 同 CTA 共享 | 程序员显式管理；32 banks |
| **Global Memory** | DRAM | 所有线程 | 最大；需要 coalescing |
| **Local Memory** | DRAM（虚拟） | 线程私有 | 寄存器溢出自动放这里；性能差 |
| **Constant Memory** | DRAM（L1缓存） | 所有线程只读 | broadcast 友好 |
| **Texture Memory** | DRAM（L1缓存） | 所有线程只读 | 2D 空间局部性优化 |

---

## 6. 内存访问模式对性能的影响

### Coalesced Access（合并访问）✅
```
Warp 内 32 个线程访问连续 128B 地址 → 1 次 memory transaction
Thread 0 → addr 0
Thread 1 → addr 4  (连续 float)
...
Thread 31 → addr 124
```
→ 1 request = 1 sector = 最高效

### Uncoalesced Access（非合并访问）❌
```
Warp 内 32 个线程访问 strided/随机地址 → 多次 transaction
Thread 0 → addr 0
Thread 1 → addr 128  (stride = 128B)
...
→ 每个线程独占一个 cache line
```
→ 1 request = 32 sectors = 带宽浪费 32x

**指标**：`l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio`  
理想值 = 1，越高越差（意味着访问越不连续）

### Shared Memory Bank Conflict（Bank 冲突）❌
Shared memory 有 32 个 bank，32-bit word 轮流分配。
同一 warp 内多个线程访问同一个 bank（但不同地址）→ 串行化。

```
示例：stride = 32 floats (=4 bytes × 32)
Thread 0 → bank 0
Thread 1 → bank 0  ← 冲突！
→ 32-way conflict = 串行执行 32 次
```

**修复**：pad 一列 `float smem[ROWS][COLS + 1];`

---

## 6.5. L1TEX 内存访问模型：Sector、Request、Wavefront

理解这三个概念是分析 memory access pattern 的基础：

```
单位层次（从大到小）：
  Instruction → Request → Wavefront → Sector

  1 个内存指令（warp 执行） = 1 个 Request
  1 个 Request              = N 个 Sector  （取决于访问连续性）
  1 个 Request              = M 个 Wavefront（取决于 cache 路径约束）
  1 个 Wavefront           = L1TEX 一个周期能处理的最大工作包
```

| 单位 | 大小 | 说明 |
|------|------|------|
| Sector | 32 bytes | 缓存和内存的最小传输单位 |
| Cache line | 128 bytes = 4 sectors | L1 和 L2 的 tag 管理粒度 |
| Request | 1 warp 的内存指令 | 包含最多 32 个线程的地址 |
| Wavefront | L1TEX 处理上限/周期 | 串行化的原因：超出 wavefront 容量 |

**关键关系**：
```python
sectors_per_request = 实际 DRAM 流量倍数
wavefronts_per_request = 实际 L1TEX 周期消耗

# 完美合并访问（coalesced）：
sectors_per_request = 1   # 32 线程覆盖 1 个 sector
wavefronts_per_request = 1

# 完全非合并访问（每线程随机地址）：
sectors_per_request = 32  # 32 线程各需一个 sector
wavefronts_per_request ≥ 4  # L1TEX 需多个周期处理
```

**为什么 sectors_per_request ≠ wavefronts_per_request？**  
一次请求可能需要 4 个 sectors 但只用 1 个 wavefront（如果 4 个 sector 都在同一 cache line 内）。也可能需要 4 个 sectors 且需要 2 个 wavefront（跨不同 cache line 时）。所以 sectors/request 反映的是 **带宽浪费**，wavefronts/request 反映的是 **延迟增加**。

**NCU 对应指标**：
- Sectors/request：`l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio`（理想=1）
- 或反向表示：`smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct`（理想=100%）

---

## 7. Pipeline 执行单元

每个 SMSP 有多条 Pipeline，同时可执行不同类型指令：

| Pipeline | 名称 | 指令类型 | Metric 前缀 |
|----------|------|---------|------------|
| `fma` | Fused Multiply Add | FADD/FMUL/FFMA (FP32), FP16x2 | `smsp__pipe_fma_cycles_active` |
| `tensor` | Tensor (MMA) | HMMA/IMMA/MMA 矩阵运算 | `smsp__pipe_tensor_cycles_active` |
| `tc` | Tensor Core (Blackwell) | UTCMMA 等新指令（Blackwell 专属） | `smsp__pipe_tc_cycles_active` |
| `fp64` | Double Precision | DADD/DMUL/DFMA | `smsp__pipe_fp64_cycles_active` |
| `alu` | Integer / Logic | 整数运算、位运算、INT IMAD/IMUL | `smsp__pipe_alu_cycles_active` |
| `lsu` | Load Store Unit | Global/Local/Shared load/store/atomic | `smsp__pipe_l1tex_cycles_active` |
| `xu` (SFU) | Special Function | sin/cos/rcp/sqrt，int↔float 转换 | `smsp__pipe_xu_cycles_active` |
| `cbu` | Convergence Barrier | 分支收敛、warp-level barrier、`__syncwarp` | — |
| `adu` | Address Divergence | 分支/跳转地址散列处理，constant load | — |
| `tma` | Tensor Memory Accelerator | Hopper async 全局↔共享内存传输 | — |
| `uniform` | Uniform Data Path | 所有线程使用相同值的标量指令 | — |

**Pipeline 利用率 > 80%** → 该 pipeline 是瓶颈

**架构差异**：
- **Ampere**：FP32 和 FP16 共享 FMA Heavy/Lite 物理资源，TC 和 FP32 同时用时互相竞争
- **Hopper**：新增 `tma` pipeline（`cp.async.bulk`），可异步搬运大块数据到 shared memory
- **Blackwell**：新增独立 `tc` pipeline（UTCMMA 指令），与旧 `tensor` pipeline 并行存在

**诊断提示**：
- `cbu` 高负载 → `__syncwarp` / barrier 指令密集，考虑减少同步频率
- `adu` 高负载 → 分支路径多样，考虑数据重排减少 divergence
- `uniform` 高利用 → 说明存在 warp-uniform 的计算，通常无需优化

---

## 8. Roofline 模型

```
        ↑ GFLOPS/s
        |         /  ← Compute Roof (Peak FLOPS)
        |        /
        |       /
        |      *  ← 你的 kernel 
        |     /
        |    /  ← Memory Bandwidth Roof
        |   /
        |  /
        +----------→ Arithmetic Intensity (FLOP/byte)
              ↑
           Ridge Point = Peak FLOPS / Peak BW
```

**Arithmetic Intensity（算术密度）**：
```
AI = 总 FLOP / 总 DRAM 字节传输量
```

- AI < Ridge Point → Memory-Bound（在 Memory Roof 以下）
- AI > Ridge Point → Compute-Bound（在 Compute Roof 以下）
- 理想：靠近对应 Roof，不在两者下方空悬

**典型值（Ampere A100）**：
- FP32 Peak: ~19.5 TFLOPS
- HBM Bandwidth: ~2 TB/s  
- Ridge Point: ~9.75 FLOP/byte

---

## 9. Roofline 数据采集（ncu）

### 采集命令

```bash
# 方法一：ncu 内置 Roofline section（GUI 可直接查看图表）
ncu --section SpeedOfLight_RooflineChart \
    --section SpeedOfLight \
    --csv --log-file roofline.csv ./kernel

# 方法二：手动采集原始指标（单 pass，误差最小）
ncu --metrics \
  dram__bytes_read.sum,\
  dram__bytes_write.sum,\
  smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,\
  smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,\
  smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,\
  smsp__sass_thread_inst_executed_op_hfma_pred_on.sum,\
  sm__inst_executed_pipe_tensor_op_hmma.sum,\
  gpu__time_duration.sum \
  --csv --log-file roofline_raw.csv ./kernel
```

### FLOP 计数方法

| 路径 | 指标 | FLOP/指令 |
|------|------|---------|
| FP32 FMA | `smsp__sass_thread_inst_executed_op_ffma_pred_on.sum` | 2 |
| FP32 ADD/MUL | `smsp__sass_thread_inst_executed_op_fadd/fmul_pred_on.sum` | 1 |
| FP16 FMA | `smsp__sass_thread_inst_executed_op_hfma_pred_on.sum` | 2 |
| TC HMMA (A100 FP16) | `sm__inst_executed_pipe_tensor_op_hmma.sum` | 8192（warp 粒度，16×16×16×2） |
| 算法已知（GEMM） | 直接解析：`2 × M × N × K` | — |

### 计算 Arithmetic Intensity 和效率

```python
# Arithmetic Intensity
total_flop  = 2 * M * N * K          # 或从上表指令计数求和
dram_bytes  = dram_read + dram_write  # ncu 采集值
AI = total_flop / dram_bytes          # FLOP/byte

# 当前性能
duration_s     = gpu__time_duration_sum_ns / 1e9
achieved_gflops = total_flop / duration_s / 1e9

# 理论上界（取 Memory Roof 和 Compute Roof 中较小值）
roof = min(AI * peak_bw_gbs, peak_gflops)

# 效率
efficiency = achieved_gflops / roof   # 目标 >70%，>85% 停止优化
```

### 多 GPU 硬件峰值

| GPU | FP32 非TC | BF16 TC | 显存 BW | Ridge(FP32) | Ridge(TC) |
|-----|----------|--------|---------|------------|----------|
| A100 SXM4 | 19.5 T | 312 T | 2.0 TB/s | 9.75 F/B | 156 F/B |
| H100 SXM5 | 67 T | 989 T | 3.35 TB/s | 20 F/B | 295 F/B |
| RTX 4090 | 82.6 T | 165 T | 1.0 TB/s | 82 F/B | 165 F/B |
| RTX 3090 | 35.6 T | 71 T | 0.94 TB/s | 38 F/B | 75 F/B |

**教学说明**：

Roofline 最大的价值是**告诉你何时该停止优化**。如果 kernel 的 Arithmetic Intensity 是 5 FLOP/byte，无论怎么优化计算，它的理论上界永远是 `5 × 2TB/s = 10 TFLOPS`——提高 TC 利用率对它毫无帮助，因为它注定是 Memory-Bound。这时唯一能突破上界的方法是**提高 AI**（即减少 DRAM 访问，比如 tiling 增加数据复用）。

---

## 10. PM 采样时间线（ncu PmSampling）

Nsight Compute 的 PM sampling 以固定时间间隔（B200 约 2µs）采样 SM 吞吐率，提供 kernel 生命周期的时间切片视图。这是唯一能看到 kernel **内部**时间分布的方式。

```bash
ncu --set full --section PmSampling --section PmSampling_WarpStates \
    -k "regex:my_kernel" -c 1 -o profile.ncu-rep ./binary
```

### 四种典型时间线形态

```
形态 1 — 平顶陡降（理想）：
  ████████████████████████████▏
  表示：SM 吞吐率高且均匀，最后一轮全部完成
  含义：Grid 充足，工作量均衡，无明显 tail effect

形态 2 — 平顶渐降（可变长度 Tail Effect）：
  ████████████████████▓▓▒▒░░  
  表示：末尾 SM 利用率逐渐降低
  含义：部分 CTA（长序列）仍在运行，其他 CTA 已完成
  根因：批次内序列长度不均，导致各 CTA 工作量差异大
  修复：Sorted Batching / Split-K / Persistent Kernel

形态 3 — 持续低位（Grid 不足 / 严重 stall）：
  ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒  
  表示：SM 全程利用率低
  含义：blocks 数 < SM 数（某些 SM 闲置），或 stall 太严重
  修复：增大 grid；检查 occupancy limiter

形态 4 — 锯齿波动（无 compute-memory overlap）：
  █░█░█░█░█░█░█░█░█░  
  表示：高算力区间 ↔ 高内存区间交替
  含义：单缓冲：先 load，等待完成，再 compute，循环往复
  修复：Double buffering（OPT-SYNC-01）/ async copy
```

### 可变长度 Tail Effect 的物理含义

假设 batch 中有 [128, 512, 2048] 三个序列长度，每个序列一个 CTA：
- CTA-0（128 token）：10ms 完成
- CTA-1（512 token）：40ms 完成  
- CTA-2（2048 token）：160ms 完成

总 kernel 时间 = 160ms（由最慢 CTA 决定）。CTA-0 和 CTA-1 完成后，对应 SM 就**闲置**了，这就形成了 PM timeline 的渐降尾部。

**诊断要点**：如果 `max_seq_len / avg_seq_len > 3`，可变长度 tail effect 是主要瓶颈。

---

## 附录 A：Hopper / Blackwell 新硬件机制（讲解新架构指标时必读）

### A.1 TMA（Tensor Memory Accelerator，Hopper+）

老方式（Ampere `cp.async`）：每个线程发一条异步拷贝指令，warp 集体搬一个 tile——消耗大量指令槽和寄存器（地址计算）。
TMA：**单线程**向 TMA 引擎提交一个"张量块拷贝描述符"，硬件自己完成整个 tile 的 global↔shared 搬运（含多维寻址、越界处理、swizzle）。

```
类比：cp.async 是 32 个工人各自跑腿搬砖；TMA 是填一张快递单，物流公司整托盘送达。
指标影响：LSU 指令数大幅下降；等待从 long_scoreboard 变为 barrier 类
        （mbarrier 等待，stall 表现为 `barrier`/`wait` 而非 long_scoreboard）。
```

### A.2 wgmma（Warp Group MMA，Hopper）与 tcgen05（Blackwell）

- 老 `mma`：单 warp 同步执行，操作数必须在寄存器 → 寄存器压力大。
- `wgmma`：4 个 warp（warp group）协同，**异步**执行，操作数可直接来自 shared memory → 计算与加载真正重叠。H100 上不用 wgmma 通常只能摸到 TC 峰值的 ~60%。
- `tcgen05`（Blackwell）：引入 **TMEM（Tensor Memory）**——TC 专属的片上存储，累加器从寄存器文件搬到 TMEM，寄存器压力进一步释放。

### A.3 Thread Block Cluster + 分布式 Shared Memory（Hopper+）

Cluster 让多个 CTA 保证同时调度在同一 GPC，可以直接读写**彼此的 shared memory**（DSMEM），打破"CTA 间只能走 global memory"的老规则。用途：跨 CTA 的 tile 复用（如 GEMM 的 split-K 归约不再走 L2）。

### A.4 对诊断的影响速记

| 现象 | 老架构解读 | Hopper+ 修正 |
|------|-----------|-------------|
| occupancy 很低（<25%） | 危险信号 | wgmma kernel 常态（大 tile + 单 CTA/SM 是策略，看 P15） |
| barrier stall 高 | __syncthreads 过多 | 可能是 TMA mbarrier 等待 = 实质上的内存延迟 |
| LSU pipe 利用率低 | 内存指令少 | TMA 把搬运卸载给专用引擎，LSU 低是好事 |
| limiter=barriers | 罕见 | TMA/mbarrier kernel 常见，减少 named barrier 数量 |

---

## 附录 B：Triton 编译管线 ↔ 硬件映射（理解"调 num_stages 为什么有效"）

```
Triton 源码                     硬件含义
─────────────────────────────────────────────────────────────
BLOCK_M/N/K        →  tile 大小 → smem 用量、寄存器压力、grid 大小、数据复用率
num_warps          →  CTA 内 warp 数（block size = num_warps×32）→ occupancy、每 warp 寄存器配额
num_stages         →  软件流水线深度 → 生成 cp.async(sm_80)/TMA(sm_90) 多级预取，
                      smem 占用 ×stages
tl.load/tl.store   →  编译器自动向量化 + 合并；但访问模式由你的指针算术决定
                      （连续维度必须由 tl.arange 的最后一维驱动，否则 sectors/req 爆炸）
tl.dot             →  mma/wgmma 指令（TC）；手写乘加循环只会生成 FFMA（P8 的常见根因）
grid               →  cdiv(M,BM)×cdiv(N,BN) → waves、tail effect
─────────────────────────────────────────────────────────────
```

**因果链示例**（把指标和参数连起来讲）：
`num_stages 2→4` ⇒ 预取窗口加深 ⇒ load 延迟被计算覆盖 ⇒ `long_scoreboard`↓ ⇒
但 smem ×2 ⇒ 可能触发 `launch__occupancy_limit_shared_mem` ⇒ occupancy↓。
**这就是为什么一次只能动一个参数，并且改完必须重新 profile。**

<!-- v2 refine 2026-06-10：新增附录 A（Hopper/Blackwell 机制）+ 附录 B（Triton 编译管线映射） -->
