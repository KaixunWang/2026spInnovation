# CrEval 集成说明

这份文档说明两件事：

1. 你现在给出的 Aman-4-Real/CrEval 到底是什么，应该怎么用。
2. 它在本项目里应该以什么形式接入，才能不破坏已有论文与结果。

## 1. CrEval 是什么

你这次给出的仓库与官网是：

- GitHub: <https://github.com/Aman-4-Real/CrEval>
- 官网: <https://creval-creative-evaluation.github.io/>

这个版本的 CrEval 是文本创造力评估模型，不是图像评测器。

根据其 README、官网和 inference.py：

- 它是一个 pairwise creativity evaluator。
- 输入不是单条文本绝对打分，而是：
  - 一条 Query / 任务指令
  - 两个候选回复 Response 1 与 Response 2
- 输出是三选一：
  - Response 1 更有创意
  - Response 2 更有创意
  - 二者创意程度相当

因此，CrEval 不是当前 src/judge.py 那种绝对维度打分器，而是一条新的外部 pairwise 评分通道。

## 2. 这意味着什么

对于你当前这篇论文，最重要的不是“训练 CrEval”，而是：

- 启动官方 CrEval 服务。
- 用它去比较同一个 rewrite 任务下的两份结果。
- 把比较结果转成当前仓库可消费的外部分数字段 creval.score。

当前仓库里已经为此补了三类工具：

- scripts/score_existing_results_with_creval.py
  - 读取已有 *_metrics.jsonl
  - 自动把 Qwen 小模型结果和对齐的 GPT 结果做 CrEval pairwise 比较
  - 输出 *_creval.jsonl
- scripts/merge_external_eval_into_metrics.py
  - 把 *_creval.jsonl 合并回 *_metrics.jsonl
- scripts/external_eval_scale_regression.py
  - 对合并后的 creval.score 做与自动指标同口径的 T3-only 二次回归和汇总

## 3. 要不要训练

如果你现在的目标是：

- 不重跑整个实验
- 只用仓库已有结果
- 给论文补一条外部评分

那么答案是：不需要训练。

你只需要做推理，不需要微调。

CrEval 官方仓库里虽然有训练背景说明，但 README 给你的可用路径就是：

1. 准备 base model
2. 准备 CrEval adapter
3. 用 LLaMA-Factory 启动 API
4. 调用 API 做 pairwise 比较

## 4. 官方 CrEval 如何启动

### 4.1 建议单独建环境

```powershell
git clone https://github.com/Aman-4-Real/CrEval.git
cd CrEval
conda create -n creval python=3.10
conda activate creval
pip install -r requirements.txt
```

上游 README 还明确写了它依赖 LLaMA-Factory，并建议使用 llamafactory=0.9.2.dev0 对应版本。如果你的环境里执行 llamafactory-cli 失败，需要按 LLaMA-Factory 的方式补装对应版本。

### 4.2 下载模型

官方提供：

- Aman/CrEval-7b
- Aman/CrEval-14b

它们是 LoRA adapter，不是完整 base model。

所以：

- CrEval-7b 对应 base model: Qwen2.5-7B-Instruct
- CrEval-14b 对应 base model: Qwen2.5-14B-Instruct

如果你只是为了给当前论文补一条外部评分，优先建议先用 7b，资源压力更小。

### 4.3 修改 creval_api.yaml

上游仓库里的模板是：

```yaml
model_name_or_path: YOUR_PATH/Qwen2.5-7B-Instruct
adapter_name_or_path: YOUR_PATH/CrEval-7b
template: qwen
finetuning_type: lora
```

你需要把 YOUR_PATH 改成你本机的真实路径。

### 4.4 启动服务

```powershell
$env:API_PORT = "8000"
$env:CUDA_VISIBLE_DEVICES = "0"
llamafactory-cli api creval_api.yaml
```

### 4.5 官方交互式测试

```powershell
python inference.py
```

inference.py 的真实输入格式已经确认是：

- Query
- Response 1
- Response 2

并且它的输出会包含以下中文模式之一：

- 更有创意的回复是：Response 1
- 更有创意的回复是：Response 2
- 二者的创意程度相当

## 5. 它在本项目里怎么接

因为 CrEval 是 pairwise evaluator，所以在本项目中最合理的接法不是“给每条 rewrite 打一个绝对分”，而是：

- 找到同一个 rewrite 任务下的两条结果
- 让 CrEval 判定谁更有创意
- 再把结果折算为候选侧的分数

当前仓库最适合的切片是：

- T3 only

原因：

- main_qwen3_4b_metrics.jsonl / 8b / 14b
- 与 main_metrics.jsonl 中的 gen_openai_4o

在 T3 条件下可以按以下 key 精确对齐：

- source_id
- condition
- target_persona
- repeat_idx
- prompt_variant

而 T2 因为随机 persona 采样不同，不能严格一一对齐，所以不建议作为 CrEval 主比较切片。

## 6. 当前仓库里新增的 CrEval 流程

### 6.1 环境变量

.env.example 里已经补了以下可选字段：

```env
CREVAL_API_KEY=0
CREVAL_BASE_URL=http://127.0.0.1:8000/v1
CREVAL_MODEL_NAME=meta-llama/Meta-Llama-3-8B-Instruct
```

这三个变量只被 CrEval 批量评分脚本使用。

### 6.2 生成 *_creval.jsonl

现在可以直接用：

```powershell
python scripts/score_existing_results_with_creval.py --reference-model gen_openai_4o
```

默认会处理：

- data/generated/main_qwen3_4b_metrics.jsonl
- data/generated/main_qwen3_8b_metrics.jsonl
- data/generated/main_qwen3_14b_metrics.jsonl

并生成：

- data/generated/main_qwen3_4b_creval.jsonl
- data/generated/main_qwen3_8b_creval.jsonl
- data/generated/main_qwen3_14b_creval.jsonl

其中每条记录的 creval.score 含义是：

- 1.0：候选 Qwen 输出胜过参考 GPT 输出
- 0.5：平局
- 0.0：候选 Qwen 输出输给参考 GPT 输出

### 6.3 合并回 metrics

```powershell
python scripts/merge_external_eval_into_metrics.py data/generated/main_qwen3_4b_metrics.jsonl data/generated/main_qwen3_4b_creval.jsonl --field-name creval --allow-missing
python scripts/merge_external_eval_into_metrics.py data/generated/main_qwen3_8b_metrics.jsonl data/generated/main_qwen3_8b_creval.jsonl --field-name creval --allow-missing
python scripts/merge_external_eval_into_metrics.py data/generated/main_qwen3_14b_metrics.jsonl data/generated/main_qwen3_14b_creval.jsonl --field-name creval --allow-missing
```

### 6.4 生成比较表和图

```powershell
python scripts/external_eval_scale_regression.py --field-name creval --score-key score --inputs data/generated/main_qwen3_4b_metrics.jsonl data/generated/main_qwen3_8b_metrics.jsonl data/generated/main_qwen3_14b_metrics.jsonl
```

输出：

- results/tables/creval_scale_regression.csv
- results/tables/creval_auto_compare.csv
- results/figures/scale_inverted_u_creval.png

## 7. 对论文应该怎么表述

建议把这一部分作为新增外部 pairwise 评价通道写进论文，不要替换原有自动指标或原 judge。

最稳的表述方式是：

- 原有 C_auto 与原 judge 保持不变
- CrEval 作为额外验证通道
- 它采用 pairwise 比较协议，因此当前分析主要报告“Qwen 候选相对 GPT 参考的胜率型分数”

这样不会破坏你现有论文结构，也符合 CrEval 的原始设计。

## 8. 如果你只关心“现在怎么做”

直接看这份文件：

- docs/creval_existing_results_only.md

那份文档只保留“已有结果补外部评分”的最短可执行路径。
