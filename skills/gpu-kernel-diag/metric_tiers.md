# 指标分级采集机制（Tier 0-3：由粗到细，按需升级）

> 设计原则：ncu 有 7000+ 指标名，但它是 `单元__计数器.子单元.汇总` 的组合爆炸，真实信息维度少得多。
> 一次性全采（--set full ≈ 10+ passes）既慢又会让 LLM 被无关异常值带偏。
> 正确做法：**廉价筛查 → 标准诊断 → 信号模糊时按瓶颈分支升级采集细粒度指标包 → LLM 再决策**。
> 每次升级由明确的"模糊信号"触发，并以固定 JSON 输出升级决定。

---

## 0. 四层结构总览

| Tier | 内容 | 成本 | 何时执行 |
|------|------|------|---------|
| **T0 筛查** | duration + launch 配置 + SOL 四件套 + occupancy（~10 指标） | 1-2 pass，秒级 | 每个候选 kernel 都跑 |
| **T1 标准** | 28 指标集 → P0-P15 pattern 匹配 | 2-4 pass | T0 确认值得优化后 |
| **T2 细粒度包** | 6 个按瓶颈分支的指标包（按需选 1-2 个） | 每包 +2-6 pass | T1 信号模糊时（见 §3 触发表） |
| **T3 终极手段** | source 级行定位 / PM 时间线 / nsys 系统级 / sanitizer | 高（source 可达 10-100x 运行时间） | T2 仍无法归因时 |

```bash
# T0（最廉价的初筛，判断"这个 kernel 值不值得花时间"）
ncu --metrics gpu__time_duration.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__throughput.avg.pct_of_peak_sustained_active,\
lts__throughput.avg.pct_of_peak_sustained_active,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
launch__grid_size,launch__block_size,launch__waves_per_multiprocessor \
  -k "regex:K" -c 1 --csv <cmd>
# T0 决策：efficiency≥95% → 跳过；P0 命中（grid 太小）→ 直接修 grid；否则 → T1
```

---

## 1. 28 集之外的 ncu 指标空间分类（哪些有用、哪些跳过）

按硬件单元前缀分 11 类。标记：★ = 已纳入 T2 包；◐ = 特定场景才采；✗ = 对优化决策无用，跳过。

| 类 | 前缀/家族 | 内容 | 评级 | 用途 |
|----|----------|------|------|------|
| 1 | `smsp__warp_issue_stalled_*`（全 18 种） | stall 全向量（含 misc/sleeping/dispatch_stall/selected 等 T1 没有的） | ★ SCHED 包 | T1 只采 4 个主要 stall；分散型 stall 必须看全向量 |
| 2 | `smsp__inst_executed_pipe_*`（全 pipe：alu/fma/fp16/fp64/lsu/tensor/xu/uniform/cbu/adu/tex） | 全流水线利用率 | ★ COMPUTE 包 | T1 只有 fma/tensor/fp64/lsu 四条；"compute-bound 但四条都不饱和"时找隐藏 pipe（SFU 超越函数、fp16 标量、uniform datapath） |
| 3 | `smsp__sass_inst_executed_op_*` / `smsp__sass_thread_inst_executed_op_*` | SASS 指令混合计数（ld/st/atom/red/branch/ffma/hfma/dfma…） | ★ COMPUTE/MEM 包 | 指令构成画像：atomic 密度、load/store 比、FLOP 精确计数 |
| 4 | `l1tex__t_sectors_pipe_lsu_mem_*_op_*` 按 op 拆分（ld/st/atom/red × global/local/shared） | L1 流量按操作类型分解 | ★ MEM 包 | T1 只看 global ld；st 流量大、atomic 串行化、local（=spill）流量都靠这组分清 |
| 5 | `lts__t_sectors_srcunit_*` / `lts__t_sectors_aperture_*` | L2 流量按来源（tex/crop/zrop）和地址空间（device/**sysmem**/peer）分解 | ★ MEM 包 | **aperture_sysmem ≠ 0 = kernel 在直接读写主机内存（UVM 缺页/零拷贝），延迟 100 倍**——T1 完全看不到的隐形杀手 |
| 6 | `launch__shared_mem_per_block_static/dynamic/driver`、`launch__registers_per_thread`、`sm__maximum_warps_per_active_cycle_pct` | 静态资源三分量 + 理论 occupancy | ★ OCC 包 | smem 限制时分清是静态声明、动态分配还是 driver 保留（Hopper TMA kernel driver 部分可观） |
| 7 | `dram__sectors_*`、`dram__bytes.sum.per_second`、fbpa__* | DRAM 子通道细节 | ◐ | 仅排查 DRAM 行冲突/通道不均时用，少见 |
| 8 | `nvltx__bytes.sum` / `nvlrx__bytes.sum`、`pcie__read_bytes.sum` | NVLink/PCIe 流量 | ◐ MULTI-GPU 包 | kernel 内含 peer 访问或 NVSHMEM 时才有意义；多卡训练优先用 nsys 看 |
| 9 | `smsp__pcsamp_*`（PC 采样） + SourceCounters section | 按 SASS 指令/源码行归因的 stall 样本 | ★ T3 | 行级定位的唯一手段，开销大 |
| 10 | `pm_sampling` 系列（时间线采样） | kernel 内指标随时间变化 | ★ T3 | 相位问题（锯齿/长尾）的唯一手段 |
| 11 | `gpc__cycles`/`tpc__`/`fe__`/`gr__`/`idc__`/`profiler__`、各 `.min/.max/.peak` 汇总变体 | 时钟域、前端、常量索引细节、profiler 自开销 | ✗ | 对优化决策无增量信息；`.avg`+`.sum` 已够 |

**结论**：值得补充进分级体系的是第 1-6、9、10 类——它们组成下面的 T2/T3 包。第 7、8 类按场景挂载。第 11 类明确排除（喂给 LLM 只会制造幻觉素材）。

---

## 2. T2 细粒度指标包定义（6 个）

> 用 `--section` 采集（名字跨版本稳定），解析时关注"关键读数"列出的指标。
> 多个包可合并到**一次** ncu 调用，避免反复跑。

### PACK-MEM（内存细分）
```bash
ncu --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Tables \
    --metrics lts__t_sectors_aperture_sysmem_op_read.sum,lts__t_sectors_aperture_sysmem_op_write.sum,\
smsp__sass_inst_executed_op_global_atom.sum,smsp__sass_inst_executed_op_local_ld.sum,\
smsp__sass_inst_executed_op_local_st.sum -k "regex:K" -c 1 -o pack_mem <cmd>
```
关键读数：①各级缓存表的 ld/st/atom/red 分行流量 ②sysmem sectors（UVM/零拷贝泄漏）
③local ld/st（spill 流量定量）④atomic 指令数（串行化嫌疑）⑤L2 compression 命中（如有）。
**回答的问题**："memory-bound 但 P1/P2/P3 都不像"时，流量到底花在哪一类操作上。

### PACK-SCHED（调度与 stall 全向量）
```bash
ncu --section SchedulerStats --section WarpStateStats -k "regex:K" -c 1 -o pack_sched <cmd>
```
关键读数：`smsp__warps_eligible.avg.per_cycle_active`、`smsp__issue_active.avg.per_cycle_active`、
全部 18 种 `smsp__warp_issue_stalled_*`（含 T1 缺失的 misc/dispatch_stall/sleeping/imc_miss/tex_throttle）。
**回答的问题**：underutilized 但没有单一 stall >15% —— stall 分散在哪里；eligible<1 是普遍饥饿还是间歇性。

### PACK-COMPUTE（流水线与指令混合）
```bash
ncu --section ComputeWorkloadAnalysis --section InstructionStats -k "regex:K" -c 1 -o pack_comp <cmd>
```
关键读数：全 pipe 利用率（alu/fma/fp16/fp64/lsu/tensor/**xu**/uniform/cbu）、SASS opcode 分布。
**回答的问题**：compute-bound 但 fma 和 tensor 都不饱和——饱和的是 XU（exp/sin/rsqrt 超越函数→
改 `__expf` 或查表）、ALU（整数地址计算过重→指针算术外提）还是 CBU（分支/收敛单元）。

### PACK-OCC（占用归因）
```bash
ncu --section Occupancy --section LaunchStats -k "regex:K" -c 1 -o pack_occ <cmd>
```
关键读数：`launch__occupancy_limit_{registers,shared_mem,warps,blocks,barriers}` 全集、
smem static/dynamic/driver 三分量、`launch__registers_per_thread`、理论 vs 实际 occupancy。
**回答的问题**：occupancy 低但 T1 的 limiter 读数之间打架/缺失时，精确归因到哪种资源、哪个分量。

### PACK-DIVERGE（分支与访问发散）
```bash
ncu --section SourceCounters --metrics smsp__sass_average_branch_targets_threads_uniform.pct,\
smsp__thread_inst_executed_per_inst_executed.ratio -k "regex:K" -c 1 -o pack_div <cmd>
```
关键读数：分支均匀度、thread 效率、（SourceCounters 附带的）发散最严重的前几个 branch 位置。
**回答的问题**：IPC 低且怀疑 divergence，但需要确认严重程度与位置。

### PACK-MULTIGPU（◐ 按场景挂载）
```bash
ncu --section Nvlink --metrics pcie__read_bytes.sum,pcie__write_bytes.sum -k "regex:K" -c 1 <cmd>
# 注意：多卡通信问题优先走 nsys（nccl recipe），本包仅用于 kernel 内 peer 访问
```

---

## 3. 升级触发表（"指标反映了问题但不够清晰" → 采哪个包）

> LLM 在 T1 pattern 匹配后，遇到下列信号时输出升级决定（JSON 格式见 §4），一次性采齐所需包。

| 信号 | 具体判据 | 升级动作 |
|------|---------|---------|
| **S1 内存类无法归因** | SOL=memory 但 P1/P2/P3 均不命中（sectors/req<2 且 L2 hit 正常） | PACK-MEM（查 atomic/st/local/sysmem 流量） |
| **S2 stall 分散** | SOL 全低（underutilized）但无任何 stall >15% | PACK-SCHED；若全向量仍分散 → T3 source |
| **S3 隐藏流水线** | SOL=compute 但 fma/tensor/fp64/lsu 全 <60% | PACK-COMPUTE（XU/ALU/uniform 嫌疑最大） |
| **S4 占用归因矛盾** | occ 低但 T1 limiter 读数缺失或互相矛盾 | PACK-OCC |
| **S5 疑似 divergence** | IPC<1.5 但 T1 无分支均匀度数据 | PACK-DIVERGE |
| **S6 两个 pattern 并列** | 两条 pattern 同时命中且行动冲突 | 同时采两条对应的包（一次 ncu 跑完） |
| **S7 数据质量可疑** | hit rate >100%、超额比 <1 但 mem SOL 高、两次采集差 >10% | 不升级，**重测**：`--clock-control base -c 8` 取中位数 |
| **S8 知道 stall 类型不知道哪行** | T2 已归因 stall 但代码中有多个候选位置 | T3：`-lineinfo` 重编译 + `--set source --import-source on`，读 source 页 per-line stall |
| **S9 相位/时变问题** | 锯齿波动、长尾、前后段行为不同的迹象 | T3：`--section PmSampling --section PmSampling_WarpStates` |
| **S10 kernel 外问题** | T0 就发现 kernel 都很快但端到端慢；或 P11 命中 | T3：转 nsys（规则 8），ncu 停止加码 |
| **S11 数值可疑** | 优化后误差增大、结果不稳定 | T3：`compute-sanitizer --tool memcheck/racecheck`（与性能无关也要跑） |

**成本警示（LLM 决策时必须考虑）**：
- 每加一个 section ≈ +2-6 个 replay pass；多包合并一次跑远优于分次跑。
- T3 source（SASS patching）开销 10-100x 运行时间，确认 S8 成立才用，且 `-c 1` 限定单实例。
- 升级链最多两跳（T1→T2→T3）。两跳后仍无法归因 → 按 agent_loop.md §8 S5 上报人工，不要无限采集。

---

## 4. 升级决定的 JSON 格式（LLM 输出，外层脚本执行）

```json
{
  "decision": "escalate",
  "from_tier": 1,
  "triggered_signals": ["S1", "S5"],
  "reason": "memory-bound 但 sectors/req=1.3、L2 hit=71%，P1/P2/P3 均不命中；同时 IPC=1.2 缺分支数据",
  "packs": ["PACK-MEM", "PACK-DIVERGE"],
  "command": "ncu --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Tables --section SourceCounters --metrics lts__t_sectors_aperture_sysmem_op_read.sum,smsp__sass_inst_executed_op_global_atom.sum,smsp__sass_average_branch_targets_threads_uniform.pct -k regex:my_kernel -c 1 -o t2_r3 <cmd>",
  "expected_discrimination": "若 atom.sum 高 → 原子串行化路径；若 sysmem>0 → UVM 泄漏路径；若 branch_uniform<80 → P10"
}
```

规则：①`expected_discrimination` 必填——采集前先写明"这次升级能区分哪几种假设"，防止漫无目的加指标；
②升级后回到 T1 的 pattern 匹配流程，新证据并入 BottleneckReport 的 evidence；
③本地小模型：只允许从 §3 触发表选信号、从 §2 选包，禁止自拟指标名。

---

## 5. nsys 侧的对应分级（系统级）

| Tier | 命令 | 用途 |
|------|------|------|
| N0 | `nsys stats --report cuda_gpu_kern_sum` | 热点表（kernel 候选筛选，喂给 ncu T0） |
| N1 | `nsys analyze`（expert system 6 规则） | 自动检测 memcpy/sync/gaps 反模式 |
| N2 | `--report cuda_kern_exec_sum`（queue time）、`cuda_gpu_mem_time_sum` | CPU↔GPU 协作健康度 |
| N3 | `nsys recipe gpu_gaps / cuda_gpu_kern_pace / nccl_gpu_overlap_trace / diff` | gap 根因、抖动、通信 overlap、前后对比 |

升级方向与 ncu 相反：nsys 由内向外（kernel → 进程 → 多卡），触发条件见 diag_rules.md 规则 5/8。

<!-- v1 2026-06-10：分级采集机制（T0-T3 + 6 个 T2 包 + 11 条升级信号） -->
