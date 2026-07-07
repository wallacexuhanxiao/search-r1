# GRPO BM25 Search-R1 实验报告

日期：2026-07-07

## 1. 实验设置

本次实验是 BM25 Search-R1 Agent 的主线 GRPO 训练。

核心设置：

- 基座模型：`Qwen/Qwen2.5-3B-Instruct`
- 训练方法：LoRA-GRPO
- 检索器：本地 Wikipedia BM25
- Agent 输入的 BM25 文档数：`top_k = 3`
- 最多搜索轮数：2
- 环境最大步数：3
- 验证频率：每 50 step 一次
- 总训练步数：300
- 训练从 `step100` checkpoint resume 后继续到 `step300`

已经从 GRPO 机器拉回本地的核心文件：

- 训练日志：
  - `remote_artifacts/grpo_20260707/logs/train_v2.pre_resume_step100.log`
  - `remote_artifacts/grpo_20260707/logs/train_v2_resume_from100.log`
- 检索命中率审计：
  - `remote_artifacts/grpo_20260707/logs/retrieval_recall_audit_validation_summary.json`
  - `remote_artifacts/grpo_20260707/logs/retrieval_recall_audit_validation.csv`
- 远端核心代码快照：`remote_artifacts/grpo_20260707/repo_core/`

最终状态：

- `train_v2_resume_from100.exitcode = 0`
- `global_step_300` 已保存
- `latest_checkpointed_iteration.txt = 300`
- 训练进度到达 `300/300`

## 2. 验证结果

| Step | NQ EM | HotpotQA EM | Success Rate | NQ 平均搜索次数 | HotpotQA 平均搜索次数 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.168 | 0.292 | 0.261 | 0.966 | 1.250 |
| 50 | 0.169 | 0.299 | 0.274 | 1.000 | 1.230 |
| 100 | 0.169 | 0.313 | 0.268 | 1.070 | 1.068 |
| 150 | 0.156 | 0.311 | 0.269 | 1.392 | 1.328 |
| 200 | 0.164 | 0.336 | 0.278 | 1.278 | 1.270 |
| 250 | **0.190** | **0.367** | **0.303** | 1.254 | 1.230 |
| 300 | 0.173 | 0.347 | 0.285 | 1.302 | 1.250 |

按指标选择：

- 主推荐 checkpoint：`step250`
- NQ 最优：`step250`，NQ EM = `0.190`
- HotpotQA 最优：`step250`，HotpotQA EM = `0.367`
- 总体 success rate 最优：`step250`，success rate = `0.303`
- 最终 checkpoint：`step300`，NQ EM = `0.173`，HotpotQA EM = `0.347`

## 3. 结果解读

GRPO 的主线结果是有效的，尤其是在 HotpotQA 上明显变强。

相对 step0：

- NQ EM 从 `0.168` 到 step250 的 `0.190`，提升 `+0.022`。
- HotpotQA EM 从 `0.292` 到 step250 的 `0.367`，提升 `+0.075`。
- Success rate 从 `0.261` 到 step250 的 `0.303`，提升 `+0.042`。

这说明 GRPO 训练确实学到了更适合多跳检索问答的行为，而不是只是在格式上过拟合。

step300 相比 step250 有回落：

- NQ EM：`0.190 -> 0.173`
- HotpotQA EM：`0.367 -> 0.347`
- Success rate：`0.303 -> 0.285`

因此最终报告和 Demo 不建议直接用 step300，而应该用 `step250` 作为主 checkpoint。step300 可以作为“训练完成 checkpoint”保留，但不是最佳模型。

## 4. 工具调用行为

GRPO 保持了较高的搜索使用率，和项目目标一致。

在最佳 step250：

- NQ 平均搜索次数：`1.254`
- HotpotQA 平均搜索次数：`1.230`

这比 GiGPO 更像一个真正的多轮 BM25 Search Agent：模型没有过度退化成 direct answer，也没有完全不搜索。

step250 的 HotpotQA 提升尤其重要，因为 HotpotQA 更依赖多跳证据。GRPO 能在 HotpotQA 上达到 `0.367`，说明它学到的搜索策略对多跳问题有效。

## 5. 稳定性指标

| Step | KL Loss | Entropy Loss | PG Clipfrac | Prompt Clip Ratio | Response Clip Ratio |
|---:|---:|---:|---:|---:|---:|
| 50 | 0.004 | 0.827 | 0.001 | 0.000 | 0.044 |
| 100 | 0.006 | 0.622 | 0.001 | 0.000 | 0.000 |
| 150 | 0.009 | 0.595 | 0.000 | 0.000 | 0.071 |
| 200 | 0.057 | 0.895 | 0.000 | 0.000 | 0.077 |
| 250 | 0.033 | 0.382 | 0.000 | 0.000 | 0.029 |
| 300 | 0.146 | 0.588 | 0.000 | 0.000 | 0.159 |

结论：

- Prompt 没有被截断：所有验证点 `prompt clip ratio = 0.000`。
- step300 的 response clip ratio 升到 `0.159`，可能是后期回落的一个信号。
- KL 在 step300 升到 `0.146`，仍不算爆炸，但明显高于 step250 的 `0.033`。
- PG clip fraction 基本为 0，没有看到 PPO clip 大面积触发。

因此最稳妥的选择仍然是 step250：指标最高，KL 更低，response 截断更少。

## 6. BM25 检索命中率

对 validation 集使用原问题做 BM25 检索审计，结果如下：

| Dataset | Total | Hit@1 | Hit@3 | Hit@5 | Empty Rate |
|---|---:|---:|---:|---:|---:|
| All | 1000 | 0.288 | 0.416 | 0.495 | 0.000 |
| NQ | 500 | 0.242 | 0.380 | 0.482 | 0.000 |
| HotpotQA | 500 | 0.334 | 0.452 | 0.508 | 0.000 |

延迟：

- 平均检索延迟：`43.16 ms`
- P95 检索延迟：`108.38 ms`

解释：

- BM25 环境可用，空结果率为 0。
- 但 `top_k=3` 的 answer string hit 只有 overall `0.416`，说明检索质量是一个真实瓶颈。
- 这能解释为什么模型即使学会搜索，最终 EM 仍然受限。
- 这个分析对项目很加分，因为它把失败拆成了“检索没找到证据”和“找到证据但推理失败”两类。

## 7. 与 GiGPO 对比

| 方法 | Checkpoint | NQ EM | HotpotQA EM | Success Rate | NQ 平均搜索次数 | HotpotQA 平均搜索次数 |
|---|---:|---:|---:|---:|---:|---:|
| GRPO | step250 | 0.190 | **0.367** | **0.303** | 1.254 | 1.230 |
| GRPO | step300 | 0.173 | 0.347 | 0.285 | 1.302 | 1.250 |
| GiGPO | step150 | **0.212** | 0.274 | 0.263 | 0.654 | 0.722 |
| GiGPO | step300 | 0.191 | 0.299 | 0.267 | 0.738 | 1.026 |

结论：

- GiGPO 在 NQ 上有更高单点峰值。
- GRPO 在 HotpotQA 和 overall success rate 上明显更强。
- GRPO 的搜索次数更高，更符合“多轮检索 Agent”的项目目标。
- 主线结果应使用 GRPO step250，GiGPO 作为对照实验。

## 8. 推荐写法

推荐最终项目中这样写：

> The GRPO-trained BM25 Search Agent achieved the best overall validation performance at step 250, improving HotpotQA EM from 0.292 to 0.367 and overall success rate from 0.261 to 0.303. Compared with the GiGPO-style variant, GRPO maintained a higher search rate and performed substantially better on multi-hop HotpotQA, suggesting that sufficient tool-use exploration is important for BM25-based multi-turn retrieval agents.

中文解释：

> GRPO 主线模型在 step250 达到最佳验证表现，将 HotpotQA EM 从 0.292 提升到 0.367，总体 success rate 从 0.261 提升到 0.303。相比 GiGPO，GRPO 保持了更高的搜索调用次数，并在多跳 HotpotQA 上明显更强，说明对 BM25 多轮检索 Agent 来说，保持足够的工具调用探索非常关键。

## 9. 下一步

建议下一步做：

1. 用 GRPO step250 跑最终 test。
2. 同时跑 Direct Answer、One-shot BM25 RAG、Prompted BM25 Search baseline。
3. 补成功/失败样本的 retrieval-hit 拆解。
4. 导出三张图：
   - Validation EM vs step
   - Success rate vs step
   - Average search turns vs step
5. Demo 默认加载 GRPO step250，GiGPO 作为 ablation 展示。
