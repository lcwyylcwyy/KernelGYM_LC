# 优化策略数据库

> 按瓶颈类型索引，每条策略含代码片段和预期收益

---

## OPT-MEM-01：Shared Memory Tiling（消除 Global Memory 延迟）

**适用**：`long_scoreboard` stall 高；L1 hit rate 低；DRAM 带宽受限  
**原理**：将频繁访问的数据加载到 shared memory，后续访问走 L1 而非 DRAM

```cuda
// ❌ Before：每次直接访问 Global Memory
__global__ void matmul(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    for (int k = 0; k < N; k++)
        sum += A[row * N + k] * B[k * N + col];  // 每次都打 Global Memory
    C[row * N + col] = sum;
}

// ✅ After：Tiling + Shared Memory
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
**预期收益**：通常 2-10x 加速，取决于数据复用程度

---

## OPT-MEM-02：Bank Conflict 消除（Shared Memory Padding）

**适用**：`l1tex__data_bank_conflicts_pipe_lsu` 高  
**原理**：Shared memory 有 32 banks，padding 打破 stride 对齐

```cuda
// ❌ Before：stride=32 导致所有线程打 bank 0
__shared__ float smem[32][32];      // col=32 → stride 32 words → all bank 0
float val = smem[threadIdx.y][threadIdx.x * 32]; // 32-way conflict

// ✅ After：+1 padding 错开 bank 映射
__shared__ float smem[32][33];      // col=33 → stride 33 words → 无冲突
float val = smem[threadIdx.y][threadIdx.x];

// 也可动态判断
#define SMEM_COLS (32 + 1)  // padding 1 element per row
__shared__ float smem[32][SMEM_COLS];
```
**预期收益**：消除 bank conflict 可提速 10-50%（取决于 conflict 严重程度）

---

## OPT-MEM-03：Global Memory Coalescing（访问对齐）

**适用**：`sectors_per_request` 高；DRAM 带宽利用率高但数据量小  
**原理**：同一 warp 的线程访问连续内存 → 1 次 transaction

```cuda
// ❌ Before：AoS（Array of Structures）— 列访问不连续
struct Particle { float x, y, z, w; };
Particle* particles;  // [x0,y0,z0,w0, x1,y1,z1,w1, ...]
float x = particles[tid].x;  // stride = 4 floats → 差 coalescing

// ✅ After：SoA（Structure of Arrays）— 行访问连续
float* xs, *ys, *zs, *ws;  // [x0,x1,x2,...], [y0,y1,y2,...], ...
float x = xs[tid];           // 连续访问 → 完美 coalescing

// ✅ 也可以用 __ldg() 启用只读缓存
float x = __ldg(&xs[tid]);  // 通过 texture cache 访问，对只读数据更好
```
**预期收益**：非合并 → 合并可提速 8-32x

---

## OPT-MEM-04：L2 Persistent Cache（Ampere+）

**适用**：working set 适度（< L2 大小），需要跨 kernel 复用的参数或权重  
**适用 GPU**：Ampere（A100）及以后

```cuda
// 将频繁访问的数据固定在 L2 中（最高 40MB on A100）
cudaStreamAttrValue stream_attribute;
stream_attribute.accessPolicyWindow.base_ptr = model_weights;
stream_attribute.accessPolicyWindow.num_bytes = weight_size;
stream_attribute.accessPolicyWindow.hitRatio = 1.0f;  // 100% 尝试驻留
stream_attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
stream_attribute.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);

// 运行 kernel...

// 使用完后重置
stream_attribute.accessPolicyWindow.num_bytes = 0;
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
cudaCtxResetPersistingL2Cache();
```
**预期收益**：参数频繁重用时 20-50% 加速

---

## OPT-OCC-01：__launch_bounds__ 控制寄存器（提升 Occupancy）

**适用**：`launch__occupancy_limit_registers` 是 occupancy 限制  
**原理**：告知编译器每个 SM 至少运行 N 个 block，编译器据此压缩寄存器使用

```cuda
// 告知编译器：最大 256 threads/block，每 SM 至少 2 个 block
__global__ __launch_bounds__(256, 2)
void my_kernel(float* data) {
    // 编译器会尽量把寄存器用量压到允许 2 blocks 并发的水平
}

// 更精确控制：配合 occupancy API
// 先查询：
int min_grid, block_size;
cudaOccupancyMaxPotentialBlockSize(&min_grid, &block_size, my_kernel, 0, 0);
// 然后用 block_size 启动
my_kernel<<<(N + block_size - 1) / block_size, block_size>>>(data);
```
**预期收益**：occupancy 从 25% 提升到 50%+ 可带来 1.5-2x 加速（当 latency-bound 时）

---

## OPT-OCC-02：Block Size 优化

**适用**：`launch__occupancy_limit_warps` 是限制；Block size < 128  
**规则**：Block size 是 32 的倍数（避免浪费 warp），最少 128 threads

```cuda
// ❌ Before：Block size 32 → 每 SM 只有 1 warp/block
dim3 block(32, 1, 1);
dim3 grid((N + 31) / 32, 1, 1);

// ✅ After：Block size 256 → 更多 warp 并发
dim3 block(256, 1, 1);
dim3 grid((N + 255) / 256, 1, 1);

// 对于 2D kernel（如 matmul）
dim3 block(16, 16, 1);    // = 256 threads，好
// 而不是
dim3 block(32, 1, 1);     // = 32 threads，太少
```
**预期收益**：低 occupancy 时 1.5-3x

---

## OPT-OCC-03：减少 Shared Memory 用量

**适用**：`launch__occupancy_limit_shared_mem` 是限制

```cuda
// ❌ Before：固定分配大块 shared memory
__global__ void kernel() {
    __shared__ float smem[1024][64];  // = 256KB，太大
}

// ✅ After：动态分配，运行时控制大小
__global__ void kernel(float* smem_base) {
    extern __shared__ float smem[];  // 动态 shared memory
    // 实际用量由 launch 时的第三个参数控制
}

// Launch 时指定实际需要的大小
size_t smem_size = actual_tile_size * sizeof(float);
kernel<<<grid, block, smem_size>>>(data);

// 另：减少精度
__shared__ __half smem[1024];  // FP16 代替 FP32，省一半空间
```
**预期收益**：occupancy 翻倍时 1.2-2x

---

## OPT-SYNC-01：减少 __syncthreads（降低 Barrier Stall）

**适用**：`wait` stall 占比高；`__syncthreads()` 调用频繁

```cuda
// ❌ Before：每次 tiling 都用 2 次 __syncthreads
for (int t = 0; t < N/TILE; t++) {
    load_tile_A();
    load_tile_B();
    __syncthreads();   // 等待加载完成
    compute();
    __syncthreads();   // 等待计算完成再覆盖 smem
}

// ✅ After：Pipeline 掩盖延迟（double buffering）
__shared__ float smem_A[2][TILE][TILE];  // 双缓冲
__shared__ float smem_B[2][TILE][TILE];
int ping = 0, pong = 1;
// 预加载第 0 块
load_tile(smem_A[ping], smem_B[ping], 0);
__syncthreads();
for (int t = 1; t < N/TILE; t++) {
    // 计算 ping 的同时，后台加载 pong（需要 async copy 支持）
    load_tile_async(smem_A[pong], smem_B[pong], t);
    compute(smem_A[ping], smem_B[ping]);
    __syncthreads();
    swap(ping, pong);
}
compute(smem_A[ping], smem_B[ping]);

// ✅ Warp 内同步用 __syncwarp() 代替 __syncthreads()
// __syncthreads() → 同步整个 block（所有 warp）
// __syncwarp()    → 只同步当前 warp（快得多）
__syncwarp(0xFFFFFFFF);  // 同步 warp 内所有活跃线程
```
**预期收益**：barrier 开销高时 10-30%

---

## OPT-COMP-01：FP16 / BF16 代替 FP32

**适用**：Compute-Bound；FP32 pipeline 饱和；精度可接受  
**硬件**：Volta+ 原生 FP16，Ampere+ 原生 BF16

```cuda
// ❌ Before：FP32
float a = ..., b = ...;
float c = a * b + a;

// ✅ After：FP16（速度 2x，带宽 2x）
#include <cuda_fp16.h>
__half a = ..., b = ...;
__half c = __hfma(a, b, a);  // FP16 FMA

// ✅ 更好：FP16 vectorized（同时处理 2 个）
__half2 a2 = ..., b2 = ...;
__half2 c2 = __hfma2(a2, b2, a2);  // 等效 2x throughput

// PyTorch 中：
# model.half()  →  全部参数转 FP16
# torch.autocast(device_type='cuda', dtype=torch.bfloat16)  →  混合精度
```
**预期收益**：FP32 → FP16 理论 2x 计算，2x 带宽

---

## OPT-COMP-02：Kernel Fusion（减少 Launch 开销和内存往返）

**适用**：nsys 中看到多个小 kernel 串行；每个 kernel 时间 < 100µs

```cuda
// ❌ Before：多个 element-wise 操作分多个 kernel
relu<<<grid, block>>>(x, y);        // kernel 1：读 x，写 y（DRAM 往返）
add<<<grid, block>>>(y, bias, z);   // kernel 2：读 y+bias，写 z（再次 DRAM 往返）
sigmoid<<<grid, block>>>(z, out);   // kernel 3：读 z，写 out

// ✅ After：融合成 1 个 kernel
__global__ void fused_relu_add_sigmoid(float* x, float* bias, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float val = fmaxf(x[i], 0.0f);      // relu
        val = val + bias[i];                  // add bias
        out[i] = 1.0f / (1.0f + expf(-val)); // sigmoid
        // 只有 1 次 DRAM 读（x[i], bias[i]）和 1 次写（out[i]）
    }
}

// PyTorch 中用 torch.compile 自动融合：
# model = torch.compile(model)  →  自动 kernel fusion
```
**预期收益**：element-wise 链可提速 2-5x；减少内存往返

---

## OPT-ASYNC-01：异步 Memory Copy（Overlap 数据传输和计算）

**适用**：nsys 中 H2D/D2H 传输和计算串行；数据传输占比 >20%

```cuda
// ❌ Before：同步传输 → 等传输完才计算
cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
kernel<<<grid, block>>>(d_data);

// ✅ After：Pinned Memory + Async Stream
float* h_pinned;
cudaMallocHost(&h_pinned, size);  // Pinned（Page-locked）memory，支持 async
memcpy(h_pinned, h_data, size);

cudaStream_t stream;
cudaStreamCreate(&stream);

// 异步传输（不阻塞 CPU）
cudaMemcpyAsync(d_data, h_pinned, size, cudaMemcpyHostToDevice, stream);

// 在同一 stream 中提交 kernel（自动在传输后执行）
kernel<<<grid, block, 0, stream>>>(d_data);

// 用第二个 stream 同时处理另一批数据（真正 overlap）
cudaMemcpyAsync(d_data2, h_pinned2, size, cudaMemcpyHostToDevice, stream2);
kernel<<<grid, block, 0, stream2>>>(d_data2);
```
**预期收益**：数据传输时间可以被完全隐藏 → 接近 2x

---

## OPT-GRAPH-01：CUDA Graph（消除 Launch Overhead）

**适用**：nsys 中 kernel 之间有频繁 CPU launch 开销；循环执行固定 kernel 序列

```cuda
// ❌ Before：每次迭代都有 CPU launch overhead（每次 ~5-20µs）
for (int i = 0; i < 1000; i++) {
    kernel_A<<<g, b>>>(data);
    kernel_B<<<g, b>>>(data);
    kernel_C<<<g, b>>>(data);
}

// ✅ After：CUDA Graph 捕获后重放
cudaGraph_t graph;
cudaGraphExec_t graphExec;
cudaStream_t stream;
cudaStreamCreate(&stream);

// 只需捕获一次
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
kernel_A<<<g, b, 0, stream>>>(data);
kernel_B<<<g, b, 0, stream>>>(data);
kernel_C<<<g, b, 0, stream>>>(data);
cudaStreamEndCapture(stream, &graph);
cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

// 循环执行只需 launch graph（<1µs overhead）
for (int i = 0; i < 1000; i++)
    cudaGraphLaunch(graphExec, stream);

cudaStreamSynchronize(stream);
```
**预期收益**：小 kernel 密集场景可提速 30-100%

---

---

## OPT-TAIL-01：Tail Effect 修复（可变长度 Batch 负载均衡）

**适用**：PM timeline 末尾渐降（非陡降）；`--page details` 报告 SM 活跃周期差异大；  
输入含不同长度的序列（attention、RNN、irregular sparse）

**原理**：CTA 的内层循环次数由输入长度决定，长序列 CTA 让短序列 CTA 闲等

```python
# 修复一：Sorted Batching（应用层，最简单）
# 按序列长度排序，让同一 wave 的 CTA 工作量相近
sequences.sort(key=lambda s: len(s), reverse=True)  # 降序，最长先处理

# 修复二：Split-K（把长序列拆给更多 CTA）
# 原始：1 CTA 处理 seq_len 个 token
# 优化：split_factor 个 CTA 各处理 seq_len/split_factor 个 token，最后 reduce
# 伪代码：
block_id = blockIdx.x
split_id = blockIdx.y   # 新增 split 维度
chunk_start = split_id * chunk_size
chunk_end   = min(chunk_start + chunk_size, seq_len)
# 各 CTA 只处理 [chunk_start, chunk_end) 的 token

# 修复三：Persistent Kernel（work-stealing）
__global__ void persistent_kernel(int* work_queue, int* queue_size) {
    while (true) {
        int task = atomicAdd(queue_size, -1) - 1;  // 动态领取任务
        if (task < 0) break;
        process_task(work_queue[task]);             // 处理一个工作单元
    }
}
```
**预期收益**：可变长度场景 1.5-3x（视长度分布方差）

---

## OPT-SPILL-01：消除寄存器溢出（Register Spill）

**适用**：`smsp__sass_inst_executed_op_local_ld.sum > 0`；`launch__registers_per_thread > 128`；  
NCU 提示 "N bytes spilled to local memory"

**原理**：编译器无法将所有变量放入寄存器，溢出到 local memory（DRAM 支撑，~800 cycles）

```cuda
// 方法一：__launch_bounds__ 告知编译器寄存器预算
// 编译器会优化掉溢出，代价是可能降低计划内并行度
__launch_bounds__(256, 4)   // maxThreadsPerBlock=256, minBlocksPerSM=4
__global__ void my_kernel(...) { ... }

// 方法二：减少中间变量，重新计算替代存储
// ❌ Before：缓存大量中间结果
float cache[32];  // 32 个 float → 32 个寄存器
for (int i = 0; i < 32; i++) cache[i] = heavy_compute(i);
for (int i = 0; i < 32; i++) use(cache[i]);

// ✅ After：即用即算（如果计算代价 << 寄存器溢出代价）
for (int i = 0; i < 32; i++) use(heavy_compute(i));

// 方法三：将 per-thread 数组移到 shared memory
// ❌ Before（溢出到 local memory）
float arr[64];     // 64 float/thread = 每 SM 256KB+ → 溢出

// ✅ After（shared memory，片上访问）
__shared__ float arr[256][64];  // 256 threads 共享，按 threadIdx 索引
arr[threadIdx.x][i] = ...;
```
**预期收益**：消除溢出可提速 2-5x（视溢出频率）

---

## OPT-FP64-01：消除意外 FP64（浮点字面量检查）

**适用**：`sm__pipe_fp64_cycles_active > 0%`，但 kernel 设计上应为全 FP32

**原理**：C/C++ 浮点字面量默认为 `double`，FP64 pipeline 吞吐量远低于 FP32

```cuda
// ❌ Before：隐式 FP64
float x = a + 1.0 * b;      // 1.0 是 double，整个表达式提升为 double
float y = sin(x);            // sin() 默认 FP64 版本
float z = 3.14159265358979;  // double 字面量

// ✅ After：显式 FP32
float x = a + 1.0f * b;     // f 后缀 → float
float y = sinf(x);           // sinf/cosf/expf/logf 等是 FP32 版本
float z = 3.14159265358979f; // f 后缀

// 常用 FP32 数学函数
// __sinf, __cosf, __expf, __logf  — 快速近似版（精度略低，速度更快）
// sinf, cosf, expf, logf          — IEEE FP32 精确版
```

```bash
# 用 nvcc 检查是否有 FP64 指令：
cuobjdump --dump-sass your_binary | grep -i "DFMA\|DADD\|DMUL\|D2F\|F2D"
# 有输出 → 确认存在 FP64 指令
```
**预期收益**：A100 上 FP32:FP64 算力比 = 2:1，消除意外 FP64 可得 ~2x compute 提速

---

## 快速索引

| 症状 | 推荐优化 | 代码 |
|------|---------|------|
| DRAM 带宽 >80% | OPT-MEM-01（Tiling） | ✅ |
| Bank conflict | OPT-MEM-02（Padding） | ✅ |
| Sectors/request >4 | OPT-MEM-03（SoA） | ✅ |
| 寄存器限制 occupancy | OPT-OCC-01（launch_bounds） | ✅ |
| Block size < 128 | OPT-OCC-02（增大 block） | ✅ |
| smem 限制 occupancy | OPT-OCC-03（动态 smem） | ✅ |
| wait stall >15% | OPT-SYNC-01（double buf）| ✅ |
| FP32 pipeline 满 | OPT-COMP-01（FP16） | ✅ |
| 多个小 kernel | OPT-COMP-02（Fusion） | ✅ |
| H2D 传输串行 | OPT-ASYNC-01（async） | ✅ |
| Launch overhead 高 | OPT-GRAPH-01（Graph） | ✅ |
| PM timeline 末尾渐降 / 可变长 batch | OPT-TAIL-01（Tail Effect） | ✅ |
| local_ld/st > 0 / 寄存器溢出 | OPT-SPILL-01（launch_bounds） | ✅ |
| FP64 pipeline 非零 | OPT-FP64-01（FP32 字面量） | ✅ |
