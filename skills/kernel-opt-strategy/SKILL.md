---
name: kernel-opt-strategy
description: KernelBench Triton 优化的两阶段策略编排。阶段1（方法探索）鼓励算法级重构——数学等价化简/算子折叠/融合/自写GEMM/布局，这些是 10-154x 高倍率的唯一来源，NCU 看不见；阶段2（局部调优）在方法稳定后转入 gpu-kernel-diag 用 NCU 精修 tile/occupancy/精度。用于迭代式 kernel 生成（STTS），决定每一轮该"换方法"还是"调当前方法"。
---

# Kernel 优化两阶段策略（探索 → 调优）

## 流程总览（一个脑 + 两阶段循环，不是三步线性管线）

```
        ┌─ kernel-opt-strategy（本 skill = 总脑，每轮都跑）──────────────┐
        │     看 methods.md 6 类手段 + 当前 profiling，决定在哪个阶段     │
        └────────────────┬─────────────────────────┬───────────────────┘
                         │                         │
        ┌────────────────▼───────────┐   ┌─────────▼──────────────────┐
        │ 阶段1 EXPLORE（找结构/算法）│   │ 阶段2 EXPLOIT（调单核）     │
        │  眼 = nsys（时间线/kernel间）│   │  眼 = ncu（单核内部）       │
        │  手册 = methods.md 6 类手段  │   │  手册 = gpu-kernel-diag     │
        │  动作 = 融合/消kernel/算法化简│  │  动作 = tile/occ/coalescing │
        │  nsys 每轮采（廉价）         │   │  ncu 仅此阶段采（贵）       │
        └────────────────┬───────────┘   └─────────┬──────────────────┘
                         │ 方法稳定：≥2轮best<5%     │
                         │  + 结构最优 + 瓶颈是实现细节│
                         └───────────►───────────────┘
                         ◄─── 探索复活：exploit 到 roofline 停滞 ≥2 轮 ───
```

**三个易错点（务必记住）**：
1. **本 skill 不是"第一步"，是贯穿全程的每轮决策脑**——每轮先判相位（explore/exploit），再调用对应的眼。
2. **methods.md（HTML 方法）与 nsys 在 EXPLORE 阶段是"一起用"的**，不是先后两步：nsys 是 explore 的眼（看 kernel 数/gap/热点），methods.md 是手册（决定融合谁、用哪条算法捷径）。
3. **方向 = 由外向内**：nsys（系统/kernel 间）先定结构、定热点 → ncu（单核内部）再钻热点。绝不反过来（先 ncu 微调会锚定在错的 kernel 分解上）。

---

## 角色定位
你是 GPU kernel 优化的**总策略师**。你不直接读 NCU 指标做微调（那是 `gpu-kernel-diag` 的活）；你决定**当前这一轮该做什么层级的优化**：是该**跳到一个全新方法**（结构性重构），还是该**在当前方法上局部精修**。

**核心信念（来自 96 题实测对照 reference/inductor/生成 Triton）**：
> inductor 基线已经会做算子融合、外包 cuBLAS/cuDNN、TF32 tensor core。要赢它，**小修小补没用**。
> 全部 ≥10× 的加速（P81 154×、P78 97×、P46、P38…）**无一例外来自算法级重构**——
> 数学等价改写、算子折叠、计算图化简。**这些 NCU 永远发现不了**，因为 NCU 只能 profile
> 你已经写出来的 kernel，无法告诉你"这个 kernel 根本不该存在"。

---

## 两阶段总览

```
        ┌─────────────────────────────────────────────┐
        │  阶段 1：方法探索（EXPLORE）                  │
        │  目标：找到正确的算法/结构（哪座山最高）       │
        │  手段：methods.md 七类手段，优先高杠杆的       │
        │  NCU：仅作信息参考，禁止用它指导微调           │
        │  评判：speedup 数量级跳变 + 找到 inductor 看   │
        │        不到的等价化简/融合机会                 │
        └───────────────────┬─────────────────────────┘
                            │ 方法稳定判据满足
                            ▼
        ┌─────────────────────────────────────────────┐
        │  阶段 2：局部调优（EXPLOIT）                   │
        │  目标：把选定方法压到硬件极限（爬到山顶）       │
        │  手段：转交 gpu-kernel-diag，NCU 分层下钻      │
        │  调：tile/BLOCK/num_warps/num_stages/occupancy │
        │      /coalescing/精度/eviction                │
        │  评判：逼近 roofline / BW floor               │
        └─────────────────────────────────────────────┘
```

**关键原则**：**不要过早进入阶段 2**。NCU 的精确单核诊断会把模型**锚定在当前 kernel 分解**上（实测 p7：NCU 让模型保留 3-kernel 结构逐个调优，封顶 2.67x；无 NCU 的探索版重构成单融合 kernel，达 3.88x）。阶段 2 是收尾，不是主战场。

---

## 阶段判定（每轮开始时执行）

### 何时在阶段 1（探索）
满足任一即留在探索：
- 还没拿到正确结果（correctness=false）→ 先求对，再求快
- 当前 speedup < 该题"方法天花板"的估计（见下方杠杆表）
- 最近一轮 speedup 相对上一轮**跳变 >15%**（方法还在长，别停）
- 还有 methods.md 中**未尝试的高杠杆手段**适用于本题

### 方法稳定判据 → 转入阶段 2（调优）
**同时**满足才切换：
1. 最近 **≥2 轮 best speedup 提升 < 5%**（方法收益见顶）
2. 当前 kernel 已是**结构最优**——含两层，缺一不可：
   - **(a) 分解最优**：没有明显可融合的 round-trip、没有可化简的冗余计算、没有可折叠的算子
   - **(b) 核内算法最优**：主 kernel 内部的**算法形式**也已选对——GEMM/Conv 的实现路线（`tl.dot` implicit-GEMM vs direct-FMA 累加 vs 库调用）对**本题 shape** 是最优的，没有"换一种核内算法可能更快"的未试选项
3. 主 kernel 的瓶颈是**实现细节**（tile 不对、occupancy 低、访存不连续），而非**算法/结构**（含核内算法形式）

> 🛑 **硬规则：只要本轮 explore 分析的 Fixes 里还挂着任何"方法/结构"级动作（含核内 GEMM/Conv 形式切换 `tl.dot`↔FMA↔库、算子折叠、融合、算法化简），就 NOT 满足判据 2 → 留在 explore 把它执行掉，禁止切 exploit。** "结构最优"= Fixes 里只剩 tile/occupancy/精度这类微调动作。
>
> 实测教训（P65 Conv2d_HardSwish_ReLU）：模型写了 implicit-GEMM `tl.dot`，但 `K=C_in·KH·KW=72` pad 到 128 浪费 44% MAC、`N=64` 喂不饱 TC。explore 分析**自己开出 Fix 1 = 换 direct-FMA 累加**，但相位机误判"分解已最优"切了 exploit，去微调那个低效 tl.dot → 卡 1.73x（GPT-5.5 用 direct-FMA 拿 4.95x）。**核内 GEMM 形式选择被当成"实现细节"漏掉了——它是结构级动作。**

### 阶段 2 内的回退（探索复活）
若 NCU 调优 **≥2 轮无进展**，且报告里出现"already autotuned / at roofline / 只能减流量"——说明当前方法已到极限。**回到阶段 1**，质疑结构本身：这个 kernel 能不能被另一种方法整个替换掉、核内 GEMM 形式能不能换（tl.dot↔FMA↔库）？（p7 教训：NCU 说"conv 是黑盒只能调 dtype"，但正解是把 conv 换成自写融合 kernel。）

---

## 阶段 1 工作协议

> **黄金问题（每轮先问）**：「inductor 已经融合并外包库了，我要赢它，**靠的是它做不到的哪一类重构？**」

### Step 1：拆解计算图，找等价化简机会
读 reference 的 `forward`，逐算子问：
- 这串后处理里，有没有**恒等式坍缩**？（`clamp(min(x,0),0,1)≡0`、`x+0`、eval 下 Dropout≡identity、单元素 LSE/softmax≡identity）
- 主算子的**输出维度**有没有被后续 mean/sum/pool **立刻压掉**？→ 能不能**先化简再算**，别物化完整输出（matmul→matvec，P46）
- 有没有**线性算子可折叠**？（AvgPool 折进 Conv 权重 P38；ConvTranspose stride-1 ≡ Conv flip）
- BN/GroupNorm 在 train 模式下与前后算子有没有**互相抵消**的项？

→ 命中任一 = **最高杠杆**，优先做。这是 10-154x 的来源。

### Step 2：若无算法捷径，选结构性手段
按 methods.md 杠杆表挑**适用于本题算子类型**的手段：融合消 round-trip > 自写 GEMM/implicit-conv > channels_last/NHWC。低精度（fp16/bf16）不是独立杠杆，只在融合/自写手段内作输入+累加精度用，**输出永远 fp32**。

### Step 3：明确写出本轮"方法假设"
在 Optimization notes 里写清：**这一轮换的是什么方法、为什么它能突破上一轮的结构瓶颈、预期数量级**。不要只说"调了 BLOCK"——那是阶段 2 的事。

### Step 4：用 NCU 数据（若有）只做一件事
**确认瓶颈大类**（compute/memory/latency-bound）和**各 kernel 占比**，用来**判断哪个算子值得换方法**。**禁止**据此调 tile——那会把你拖进阶段 2。

---

## 阶段 2 工作协议

转交 `gpu-kernel-diag` skill，按其 Profile→Diagnose→Prescribe→Patch 循环走。本 skill 此时只做**边界守卫**：
- 每轮检查是否触发"探索复活"回退条件
- 确保调优**不破坏**阶段 1 选定的结构（别让微调把融合 kernel 拆回多 kernel）

---

## 与其他 skill 的关系

| Skill | 角色 | 何时 |
|-------|------|------|
| **kernel-opt-strategy**（本） | 总策略：探索 vs 调优的相位决策 | 每轮，最先 |
| `gpu-kernel-diag` | NCU 局部调优执行 | 阶段 2 |
| `gpu-kernel-analyzer` | 指标深解（辅助 diag） | 阶段 2 信号模糊时 |
| `gpu-kernel-learn` | 指标硬件机制教学 | 不进自动循环 |

<!-- v1 2026-06-13：两阶段策略，源自 fastp@1.2 96题方法对照 + p7 NCU锚定实测 -->
