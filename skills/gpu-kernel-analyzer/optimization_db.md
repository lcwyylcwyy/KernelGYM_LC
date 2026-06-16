# 优化策略数据库（综合版）

> 按瓶颈类型索引，含 CUDA 代码示例和预期收益。
> 诊断模式：直接查对应 OPT-* 条目。教学模式：含原理说明。

---

## MEMORY-BOUND 优化

### OPT-MEM-01：Shared Memory Tiling

**适用症状**：`long_scoreboard` stall 高；L1 hit rate <50%；DRAM 带宽 >70%  
**原理**：将频繁访问的数据块预取到 shared memory（L1 速度），消除 DRAM 往返

```cuda
// ❌ Before：每次乘法都打 Global Memory（~800 cycles/次）
__global__ void matmul(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    for (int k = 0; k < N; k++)
        sum += A[row * N + k] * B[k * N + col];
    C[row * N + col] = sum;
}

// ✅ After：Tiling，每次只打 DRAM 加载一个 tile，后续复用走 L1
#define TILE 32
__global__ void matmul_tiled(float* A, float* B, float* C, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;
    for (int t = 0; t < N / TILE; t++) {
        As[threadIdx.y][threadIdx.x] = A[row * N + t * TILE + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * N + col];
        __syncthreads();
        for (int k = 0; k < TILE; k++) sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    C[row * N + col] = sum;
}
```
**预期收益**：2-10× 加速，取决于数据复用率  
**教学**：每个元素从 DRAM 只读一次加载到 smem，之后 TILE 次复用都走 ~20 cycle 的 smem，代替每次 ~800 cycle 的 DRAM。

---

### OPT-MEM-02：Bank Conflict 消除（Padding）

**适用症状**：`l1tex__data_bank_conflicts_pipe_lsu.sum` 高；bank_conflict_ratio > 10%  
**原理**：Shared Memory 32 banks，padding 1 列打破 stride 对齐

```cuda
// ❌ Before：stride=32 → 所有线程打同一 bank → 32-way conflict
__shared__ float smem[32][32];

// ✅ After：+1 padding 将 bank 映射错开
__shared__ float smem[32][33];   // 每行多 1 个 float，消除 bank 冲突

// 通用宏
#define SMEM_COLS (TILE + 1)
__shared__ float smem[TILE][SMEM_COLS];
```
**预期收益**：消除严重 bank conflict 可提速 10-50%

---

### OPT-MEM-03：Global Memory Coalescing（SoA 布局）

**适用症状**：`l1tex__average_t_sectors_per_request...ratio` > 4；DRAM 利用率高但数据量小  
**原理**：同 warp 线程访问连续地址 → 1 次 transaction

```cuda
// ❌ Before：AoS — 访问 x 字段时跳 stride=4
struct Particle { float x, y, z, w; };
Particle* particles;
float x = particles[tid].x;   // stride=4 floats，poor coalescing

// ✅ After：SoA — 访问 xs 时连续
float *xs, *ys, *zs, *ws;
float x = xs[tid];             // 完全 coalesced

// ✅ 也可用只读缓存
float x = __ldg(&xs[tid]);    // texture cache，对只读数据更高效
```
**预期收益**：非合并→合并 最高 32×

---

### OPT-MEM-04：L2 Persistent Cache（Ampere+）

**适用症状**：相同数据被多个 kernel 反复从 DRAM 读取；L2 hit rate 偏低  
**适用 GPU**：A100+（最大 40MB）

```cuda
cudaStreamAttrValue attr;
attr.accessPolicyWindow.base_ptr = weights_ptr;
attr.accessPolicyWindow.num_bytes = weights_size;  // 不超过 L2 的 1/3
attr.accessPolicyWindow.hitRatio = 1.0f;
attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);

// 用完后清除
attr.accessPolicyWindow.num_bytes = 0;
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
cudaCtxResetPersistingL2Cache();
```
**预期收益**：参数/权重频繁重用时 20-50%

---

## OCCUPANCY / LATENCY-BOUND 优化

### OPT-OCC-01：`__launch_bounds__` 控制寄存器用量

**适用症状**：`launch__occupancy_limit_registers` 是限制；occupancy < 50%

```cuda
// 告知编译器：每 SM 至少跑 2 个 block，请压缩寄存器
__global__ __launch_bounds__(256, 2)
void my_kernel(float* data) { /* ... */ }

// 配合自动调优
int min_grid, block_size;
cudaOccupancyMaxPotentialBlockSize(&min_grid, &block_size, my_kernel, 0, 0);
my_kernel<<<(N + block_size - 1) / block_size, block_size>>>(data);
```
**预期收益**：occupancy 翻倍时 1.5-2×（latency-bound 场景）

---

### OPT-OCC-02：增大 Block Size

**适用症状**：`launch__occupancy_limit_warps`；block size < 128；`no_instruction` stall 高

```cuda
// ❌ Block 太小 → warp 少 → 无法隐藏延迟
dim3 block(32);
// ✅ 至少 128 threads，建议 256
dim3 block(256);

// 2D kernel 的对比
dim3 block_bad(8, 4);    // = 32 threads，太少
dim3 block_ok(16, 16);   // = 256 threads，好
```
**预期收益**：低 occupancy 时 1.5-3×

---

### OPT-OCC-03：减少静态 Shared Memory

**适用症状**：`launch__occupancy_limit_shared_mem` 是限制

```cuda
// ❌ 固定大块 smem 限制每 SM 只跑 1 个 block
__global__ void kernel() {
    __shared__ float smem[1024][64]; // = 256KB，太大
}

// ✅ 动态分配，按实际需要量运行
__global__ void kernel() {
    extern __shared__ float smem[];
}
kernel<<<grid, block, actual_size_bytes>>>(data);  // 运行时指定

// ✅ 也可降低精度
__shared__ __half smem[1024]; // FP16 省一半空间
```
**预期收益**：smem 减半 → occupancy 翻倍 → 1.2-2×

---

### OPT-SYNC-01：减少 `__syncthreads`（Double Buffering）

**适用症状**：`wait` stall 占比 >15%；频繁 `__syncthreads()` 调用

```cuda
// ✅ Double Buffering：计算当前 tile 的同时加载下一个 tile
__shared__ float A_buf[2][TILE][TILE];
__shared__ float B_buf[2][TILE][TILE];
// ping-pong 交替，减少必须等待的 sync 点数

// ✅ 范围更小时用 __syncwarp() 替代 __syncthreads()
// __syncthreads() → 等整个 block 的所有 warp
// __syncwarp()    → 只等当前 warp 内 32 线程（快得多）
if (lane_id < 16) {
    smem[threadIdx.x] += smem[threadIdx.x + 16];
    __syncwarp();  // warp 内同步就够了
}
```
**预期收益**：barrier 密集场景 10-30%

---

## COMPUTE-BOUND 优化

### OPT-COMP-01：FP16 / BF16 代替 FP32

**适用症状**：FP32 FMA 管道 >80%；精度可接受

```cuda
#include <cuda_fp16.h>

// FP16 标量
__half a, b, c;
c = __hfma(a, b, a);         // FP16 FMA

// FP16 向量化（同时处理 2 个，更高效）
__half2 a2, b2, c2;
c2 = __hfma2(a2, b2, a2);    // 2× throughput

// PyTorch 中
model.half()                   // 全精度转 FP16
# 或混合精度
with torch.autocast('cuda', dtype=torch.bfloat16):
    output = model(input)
```
**预期收益**：FP32 → FP16 理论 2× 计算速度，2× 内存带宽

---

### OPT-COMP-02：Kernel Fusion（减少 DRAM 往返）

**适用症状**：nsys 中多个小 kernel 串行；每个 kernel < 100µs；element-wise 操作链

```cuda
// ❌ 三个 kernel → 3次 DRAM 读 + 3次 DRAM 写
relu<<<g,b>>>(x, y);
add<<<g,b>>>(y, bias, z);
sigmoid<<<g,b>>>(z, out);

// ✅ 一个 kernel → 只有 2次 DRAM 读（x, bias）+ 1次写（out）
__global__ void fused_relu_add_sigmoid(float* x, float* bias, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float v = fmaxf(x[i], 0.0f) + bias[i];
        out[i] = 1.0f / (1.0f + expf(-v));
    }
}

// PyTorch 中 torch.compile 自动融合
model = torch.compile(model)
```
**预期收益**：element-wise 链 2-5×；内存带宽降低 ~3×

---

## SYSTEM-LEVEL 优化

### OPT-ASYNC-01：异步 Memory Copy（Overlap 传输和计算）

**适用症状**：nsys 中 H2D/D2H 传输与 kernel 串行；数据传输占比 >20%

```cuda
// ❌ 同步 → 等传输完才计算
cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
kernel<<<g,b>>>(d_data);

// ✅ Pinned Memory + Async Stream → 传输与计算 overlap
float* h_pinned;
cudaMallocHost(&h_pinned, size);  // Page-locked，支持 DMA

cudaStream_t s1, s2;
cudaStreamCreate(&s1); cudaStreamCreate(&s2);

// s1 传输 + 计算 batch 0，s2 同时传输 batch 1
cudaMemcpyAsync(d_data0, h_pinned0, size, cudaMemcpyHostToDevice, s1);
kernel<<<g, b, 0, s1>>>(d_data0);  // s1 中：传输完成后自动触发

cudaMemcpyAsync(d_data1, h_pinned1, size, cudaMemcpyHostToDevice, s2);
kernel<<<g, b, 0, s2>>>(d_data1);  // 与 s1 并行
```
**预期收益**：传输延迟完全隐藏 → 接近 2×

---

### OPT-GRAPH-01：CUDA Graph（消除 Launch Overhead）

**适用症状**：nsys 中 kernel 之间有频繁 CPU 开销（每次 launch ~5-20µs）；循环执行固定序列

```cuda
// ❌ 每次迭代都付 CPU launch 开销
for (int i = 0; i < 1000; i++) {
    kernel_A<<<g,b>>>(data);
    kernel_B<<<g,b>>>(data);
}

// ✅ 捕获一次，重放 1000 次
cudaGraph_t graph;
cudaGraphExec_t graphExec;
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
kernel_A<<<g, b, 0, stream>>>(data);
kernel_B<<<g, b, 0, stream>>>(data);
cudaStreamEndCapture(stream, &graph);
cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

for (int i = 0; i < 1000; i++)
    cudaGraphLaunch(graphExec, stream);  // <1µs overhead

// PyTorch 中
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    output = model(static_input)
for _ in range(1000):
    g.replay()
```
**预期收益**：小 kernel 密集场景 30-100%；Python overhead 场景更显著

---

---

## TAIL EFFECT 优化

### OPT-TAIL-01：可变长度负载均衡

**适用症状**：PM timeline 末尾渐降（长尾）；details 报 SM active cycles 差异大；batch 含不同长度序列

**原理**：变长输入导致各 CTA 工作量不同，最慢 CTA 决定整体延迟。

```python
# 方法 1：Sorted Batching（最简单，应用层）
sequences.sort(key=lambda s: len(s), reverse=True)
# 让同一 wave 的 CTA 工作量相近，消除明显 tail

# 方法 2：Split-K（将长序列拆给多个 CTA）
# 原始 grid: (batch_size,)
# 新 grid: (batch_size, split_factor)
# 每个 CTA 只处理 seq_len / split_factor 个 token，最后 atomic reduce
split_factor = max(1, seq_len // target_chunk_size)
grid = (batch_size, split_factor)

# 方法 3：Persistent Kernel（work-stealing，最灵活）
__global__ void persistent(int* task_idx, int total_tasks) {
    while (true) {
        int t = atomicAdd(task_idx, 1);
        if (t >= total_tasks) break;
        process(t);  // 每个工作单元独立，动态分配
    }
}
```
**预期收益（视分布方差）**：1.5-3x

---

## LATENCY / REGISTER 优化

### OPT-SPILL-01：消除寄存器溢出

**适用症状**：`smsp__sass_inst_executed_op_local_ld.sum > 0`；NCU 提示 "bytes spilled to local memory"；`launch__registers_per_thread > 128`

**原理**：register 溢出到 local memory → DRAM 支撑，~800 cycles/次

```cuda
// 方法 1：__launch_bounds__ 限制寄存器预算
__launch_bounds__(256, 4)   // maxThreads/block, minBlocks/SM
__global__ void my_kernel(...) { ... }

// 方法 2：重算代替缓存（如果计算代价 << 溢出代价）
// 方法 3：per-thread 数组移到 shared memory
__shared__ float smem[BLOCK_SIZE][K];  // 替代 float arr[K]（per-thread）
smem[threadIdx.x][i] = ...;
```
**预期收益**：2-5x（视溢出频率）

---

### OPT-FP64-01：消除意外 FP64

**适用症状**：`sm__pipe_fp64_cycles_active > 0%`，但 kernel 应为全 FP32

```cuda
// ❌ 常见错误：字面量默认 double
float x = a + 1.0 * b;   // 1.0 是 double
float y = sin(x);          // sin() 是 FP64

// ✅ 修复：全加 f 后缀 + FP32 函数
float x = a + 1.0f * b;
float y = sinf(x);  // 或 __sinf(x)（近似，更快）
```

```bash
# 检查 SASS 中是否有 FP64 指令
cuobjdump --dump-sass ./kernel | grep -E "DFMA|DADD|DMUL"
```
**预期收益**：A100 FP32:FP64 = 2:1，消除后 ~2x compute

---

## TRITON 优化

### OPT-TRITON-01：Autotune 搜索空间（受指标剪枝）

**适用症状**：Triton kernel 用固定 BLOCK/num_warps/num_stages；或参数明显不合理（如 elementwise BLOCK=4）

```python
import triton, triton.language as tl

@triton.autotune(
    configs=[
        # 用诊断结论剪枝：long_scoreboard 高 → 偏向大 num_stages；
        # limiter=registers → 偏向小 num_warps；grid 小 → 偏向小 BLOCK
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],          # 形状变化时重新 autotune
)
@triton.jit
def matmul_kernel(...): ...
```
**预期收益**：错误 config → 合理 config 可达数倍；TritonForge 极端案例（BLOCK=4 的 elementwise）95x  
**注意**：autotune 是兜底而非起点——全网格爆搜浪费编译时间，先用规则 10 映射表把空间剪到 ≤20 个

---

### OPT-TRITON-02：num_stages 软件流水线（隐藏 global→smem 延迟）

**适用症状**：P5/P9 命中（long_scoreboard 高 / TC 等数据）；Triton kernel

```python
# num_stages=N 让 Triton 生成 N 级流水线：第 i 个 tile 计算时，第 i+1..i+N-1 个 tile 在后台加载
# Ampere: 编译为 cp.async；Hopper+: 编译为 TMA（自动选择）
triton.Config({...}, num_stages=4)   # 从 2 开始逐级加

# 代价：smem 占用 ×N → 盯 launch__occupancy_limit_shared_mem
# 验证：重新 profile，long_scoreboard 应显著下降；若 occupancy 反而掉 → 退回或减 BLOCK_K
```
**预期收益**：memory latency 主导的 GEMM 类 20-60%

---

## HOPPER+ 专项

### OPT-HOPPER-01：TMA + wgmma（手写 CUDA 的代际升级）

**适用症状**：sm_90+；手写 CUDA kernel 用老式 ld.global+st.shared / mma；TC 利用率上不去（P9）

```cuda
// Hopper 起手式：TMA bulk copy + mbarrier 等待 + wgmma 异步矩阵乘
// ① TMA：单线程发起整 tile 拷贝（替代全 warp 搬运，省寄存器、省指令）
#include <cuda/barrier>
__shared__ alignas(128) __half smem_A[BM][BK];
__shared__ cuda::barrier<cuda::thread_scope_block> bar;

if (threadIdx.x == 0) {
    cde::cp_async_bulk_tensor_2d_global_to_shared(&smem_A, &tensor_map_A, x0, y0, bar);
}
bar.arrive_and_wait();   // mbarrier 等数据到位

// ② wgmma（warp group = 4 warps 协同，异步执行，PTX 内联或用 CUTLASS/CuTe 封装）
// 实践建议：手写 wgmma 极易出错，优先用 CUTLASS 3.x CollectiveMma 或直接写 Triton
```
**预期收益**：相对 Ampere 风格代码，H100 上 GEMM 类 kernel 1.3-2x；不用 wgmma 只能达 TC 峰值 ~60%  
**替代**：Ampere 用 `cp.async`（`__pipeline_memcpy_async`）；Triton 用户只需 num_stages（自动生成）

---

## 快速索引

| 观察到的症状 / 指标 | 推荐策略 | 优先级 |
|-----------------|---------|-------|
| `long_scoreboard` >20% | OPT-MEM-01（Tiling）+ OPT-OCC-01/02（Occupancy）| 高 |
| Bank conflict ratio >10% | OPT-MEM-02（Padding）| 高 |
| Sectors/request >4 | OPT-MEM-03（SoA）| 高 |
| DRAM 利用 >80% | OPT-MEM-01（Tiling）| 高 |
| PM timeline 末尾渐降 | OPT-TAIL-01（Sorted batching / Split-K）| 高 |
| local_ld/st > 0 / 寄存器溢出 | OPT-SPILL-01（launch_bounds）| 高 |
| Register limit occupancy | OPT-OCC-01（launch_bounds）| 中 |
| Block size <128 | OPT-OCC-02（增大 block）| 中 |
| Smem limit occupancy | OPT-OCC-03（动态 smem）| 中 |
| `wait` stall >15% | OPT-SYNC-01（double buf）| 中 |
| FP32 pipeline >80% | OPT-COMP-01（FP16）| 中 |
| FP64 pipeline 非零（应为 FP32 kernel）| OPT-FP64-01（字面量加 f）| 中 |
| 多个小 kernel 串行 | OPT-COMP-02（Fusion）| 中 |
| H2D 传输串行 | OPT-ASYNC-01（async copy）| 中 |
| CPU launch overhead | OPT-GRAPH-01（CUDA Graph）| 低→高（迭代场景）|
| Triton 参数可疑 / 未 autotune | OPT-TRITON-01（autotune）| 高 |
| Triton long_scoreboard 高 | OPT-TRITON-02（num_stages）| 高 |
| sm_90+ 手写 CUDA 用老式拷贝/mma | OPT-HOPPER-01（TMA+wgmma）| 中 |
