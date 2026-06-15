# 七类优化手段 playbook（KernelBench Level-2，对照 inductor 基线）

> 数据来源：96 题（GPT-5.5 62 + Claude-4.6 34）逐题对照 reference / inductor / 生成 Triton。
> speedup = 生成 Triton vs inductor（inductor 已含融合 + cuBLAS/cuDNN + GEMM 题 TF32）。
> **按杠杆从高到低排列。前两类是数量级加速（10-154x）的唯一来源，且 NCU 看不见。**

---

## 杠杆总表

| # | 手段 | 杠杆 | 适用算子 | 代表题 | NCU 可见? |
|---|------|------|---------|--------|----------|
| 1 | 常量/恒等坍缩捷径 | ★★★★★ (10-154x) | 任意+特定后处理组合 | P81 154× | ❌ 算法级 |
| 2 | 数学归约化简 | ★★★★★ (10-50x) | GEMM/Conv + mean/sum/LSE | P46, P78 97× | ❌ 算法级 |
| 3 | Conv/Pool 折叠 | ★★★★ (5-47x) | Conv/Linear + Pool | P38 47×, P97 | ❌ 算法级 |
| 4 | 卷积后处理融合 | ★★★ (2-6x) | Conv/ConvT + 逐元素/归约 | P91, P80 | ◐ 部分 |
| 5 | 自写 tiled/implicit GEMM | ★★★ (2-6x) | GEMM/Conv | P65, P34 | ◐ 部分 |
| 6 | channels_last / NHWC 布局 | ★★ (1.3-2x) | Conv/ConvT | P43, P92 | ✓ NCU可调 |

> **战略含义**：手段 1-3 在**阶段 1（探索）**靠**读计算图**发现，不靠 profiling。
> 手段 4-6 兼具结构选择（阶段1）与参数精修（阶段2）。手段 6 最适合留给 NCU。
>
> **🚫 不收录两类"伪优化"**：①**低精度红利**（单纯把 GEMM/epilogue 降 fp16/bf16 换吞吐）——这是精度红利非算法优化，且端到端 fp16 验证集会抹平它，还诱发 fp16-out 陷阱；fp16 只在**已有结构性手段内**作为输入/累加精度顺带用（见手段 4/5），不作为独立杠杆。②**benchmark 分布概率估算**（靠输入分布近似输出，如概率性"多半为零"）——这是 gaming benchmark 不是真优化；只允许**对任意合法输入恒成立的精确恒等坍缩**（手段 1）。

---

## 1. 常量 / 恒等坍缩捷径 ★★★★★

**识别**：整条 forward 在**精确恒等式**下**坍缩为常量或廉价 view**——对任意合法输入都成立，不依赖输入分布。
**触发模式**：
- `clamp(min(x, L), L, U) ≡ L`（任意 x，L≤U）→ P81 输出恒为 0，跳过 Conv3d+GroupNorm+Dropout，**154×**
- 整个分支在 eval/给定 flag 下不可达（如 eval 模式 Dropout ≡ identity）
- 算子链代数化简为常量（如 `x - x`、`x * 0`、`relu(-|y|) ≡ 0`）

**做法**：直接 `return` 常量或 `expand` 的零步长缓冲区，**整个计算图跳过**。
**陷阱**：必须**对所有合法输入恒成立**（数学恒等式），不是"benchmark 分布下多半成立"——后者是 gaming，禁止。先证恒等式，再写。

## 2. 数学归约化简 ★★★★★

**识别**：主算子（GEMM/Conv）的**完整输出被后续 reduction（mean/sum/logsumexp/softmax）立刻压掉**，物化完整输出是浪费。
**触发模式**：
- `mean(X @ W, dim=1)` = `X @ mean(W, dim=...)` → matmul 退化成 **matvec**（P46）；主计算从 O(B·in·out) 降到 O(B·in)
- 单元素维度上的 `logsumexp`/`softmax` ≡ identity
- `sum/mean` 与线性层交换顺序，先缩小再算

**做法**：代数重排，**只算归约后真正需要的量**，绝不物化 [B, out] 全输出。
**陷阱**：注意数值（重排可能改变累加顺序/精度）；reduction 维度退化为 1 时才有 identity。

## 3. Conv / Pool 折叠与池化归约 ★★★★

**识别**：**线性算子可吸收进另一个线性算子的权重**，或后处理可在归约窗口内就地完成。
**触发模式**：
- `AvgPool(Conv(x))` → 折成一个 `Conv(kernel扩大)`，GEMM 输出维度直接降 16×（P38）
- ConvTranspose stride-1 ≡ `Conv(flip(W).T, pad=k-1-p)`（走更快的 cuDNN conv 路径）
- 池化窗口内融合逐点后处理 + 直接归约，不物化中间张量

**做法**：预计算折叠后的权重（init 时一次），forward 走单个更小的算子。
**陷阱**：权重折叠要随 `load_state_dict` 失效重算（用 `_version` 钩子）。

## 4. 卷积后处理融合 ★★★（cuDNN conv + Triton epilogue）

**识别**：Conv/ConvT 后跟一串逐元素/池化/归约，inductor 已融了 epilogue 但仍有 round-trip。
**做法**：cuDNN 出 conv → **一个 Triton kernel** 融合所有后处理（bias+激活+pool+norm），消掉中间 conv 输出的写出-读回。fp16 conv（fp32 累加）+ fp32 out。
**陷阱**：conv bias 常与后续 BN/InstanceNorm 抵消 → 设 bias=None 省一遍流量。epilogue 是 BW-bound，到 roofline 后只能减流量（转阶段 2 用 NCU 确认）。
**🛑 红线：最终输出 dtype 必须 fp32，禁止 fp16/bf16 输出。** grader `correctness.py:92` 的 `torch.allclose(out, out_new, atol=1e-2, rtol=1e-2)` **不做 dtype cast**——fp32-ref 直接对比 fp16-out，量化误差几乎必然超 tol（实测 p7：fp16-out 经 sigmoid+bias 后 allclose 失败，整轮归零）。"fp16 输出减半写流量"是**致命诱饵**：省的那点带宽不值一次 correctness 失败。中间计算可 fp16，**写出张量永远 fp32**（`empty_like(x, dtype=torch.float32)`）。降精度只用在 conv/GEMM 的输入和 tensor-core 累加路径，绝不用在输出。

## 5. 自写 tiled / implicit-GEMM Conv ★★★

**识别**：固定 shape 下，cuBLAS/cuDNN 的通用性有富余，自写 tile 能把 epilogue 合进主循环。
**做法**：Triton M/N/K tiled GEMM 或 implicit-GEMM conv，epilogue（bias/scale/激活）在累加后就地做，省一次全量输出后处理。bf16/fp16 tensor core。
**陷阱**：牺牲通用性换固定 shape 性能；tile/warps 是阶段 2 的事，先把"自写 vs 库调"这个**结构**选对。

## 6. channels_last / NHWC 布局 ★★

**识别**：Conv/ConvT 题，NCHW 下有隐式 layout reorder pass。
**做法**：`channels_last(_3d)`，让 cuDNN 原生出 NHWC，与 Triton epilogue 布局对齐，消 reorder。单 op `.to(dtype=half, memory_format=channels_last)` 比两步省。
**陷阱**：out 须 fp32（allclose 布局无关但 dtype 敏感）。**此手段最适合留给阶段 2 用 NCU 验证 coalescing/sectors-per-request 收益**。

---

## 阶段 1 决策树（选哪个手段）

```
读 reference forward
  ├─ 有恒等坍缩/常量输出? ──────────→ 手段1（最高，先证后写）
  ├─ 完整输出被 reduction 压掉? ────→ 手段2（matvec化简）
  ├─ 有线性算子可折叠/池化可吸收? ──→ 手段3（权重折叠）
  ├─ 主算子是 Conv/ConvT?
  │     ├─ 后处理多且有 round-trip? ─→ 手段4（融合epilogue）
  │     ├─ 固定shape值得自写? ───────→ 手段5（implicit-GEMM）
  │     └─ 仅 layout 问题? ──────────→ 手段6（NHWC，可留阶段2）
  └─ 主算子是 GEMM?
        ├─ 固定shape值得自写? ───────→ 手段5（tiled GEMM，epilogue合进主循环）
        └─ 否则 ─────────────────────→ 手段4（库GEMM + 融合后处理epilogue）
```

**注**：手段 1-3 互斥优先于 4-6——能算法化简就别费力调 kernel。一题可叠加（如 手段3 折叠 + 手段4 融合）。低精度（fp16/bf16）**不是独立手段**，只在手段 4/5 内作为输入/累加精度顺带用，且**输出永远 fp32**（手段 4 红线）。
