# CrEval 对比已训练小模型（OpenRouter / DeepSeek）完整步骤

本流程只做外部评估对比，不重跑生成。

适用目标：

1. 使用现有小模型结果（Qwen3-4B/8B/14B 的 `*_metrics.jsonl`）。
2. 用 CrEval（训练过评估器）做 pairwise 创意比较。
3. 用普通未训练LLM（OpenRouter 或 DeepSeek API）做 baseline 比较。
4. 输出回归对比 + win-rate 对比表。

## 0. 需要的输入文件

默认会使用：

- `data/generated/main_qwen3_4b_metrics.jsonl`
- `data/generated/main_qwen3_8b_metrics.jsonl`
- `data/generated/main_qwen3_14b_metrics.jsonl`
- `data/generated/main_metrics.jsonl`（作为参考模型池，默认 `gen_openai_4o`）

如果你的结果在别的目录，后续命令都可用 `--inputs` 指定。

## 1. 你会调用到的评估器/API（全部列出）

### 1.1 CrEval（训练过评估器）

- 类型：本地 OpenAI-compatible API（通常由 LLaMA-Factory 启动）
- 模型：`CrEval-7b` adapter + `Qwen2.5-7B-Instruct` base
- 相关环境变量：
  - `CREVAL_API_KEY`
  - `CREVAL_BASE_URL`
  - `CREVAL_MODEL_NAME`

### 1.2 普通未训练LLM baseline（两种可选）

A. OpenRouter：

- Base URL: `https://openrouter.ai/api/v1`
- 可用模型举例：
  - `openai/gpt-4o-mini`
  - `deepseek/deepseek-chat-v3-0324`
  - `anthropic/claude-3.5-haiku`

B. DeepSeek 官方：

- Base URL: `https://api.deepseek.com`
- 可用模型举例：
  - `deepseek-chat`
  - `deepseek-reasoner`

baseline 相关环境变量：

- `PAIRWISE_EVAL_API_KEY`
- `PAIRWISE_EVAL_BASE_URL`
- `PAIRWISE_EVAL_MODEL`
- `PAIRWISE_EVAL_TIMEOUT`

## 2. .env 配置模板

### 2.1 CrEval 本地服务

```env
CREVAL_API_KEY=0
CREVAL_BASE_URL=http://127.0.0.1:8000/v1
CREVAL_MODEL_NAME=meta-llama/Meta-Llama-3-8B-Instruct
```

说明：`CREVAL_MODEL_NAME` 对本地服务通常只是请求字段占位，关键是服务端实际加载了 CrEval adapter。

### 2.2 baseline 走 OpenRouter（推荐）

```env
PAIRWISE_EVAL_API_KEY=<你的_openrouter_key>
PAIRWISE_EVAL_BASE_URL=https://openrouter.ai/api/v1
PAIRWISE_EVAL_MODEL=openai/gpt-4o-mini
PAIRWISE_EVAL_TIMEOUT=300
```

### 2.3 baseline 走 DeepSeek 官方

```env
PAIRWISE_EVAL_API_KEY=<你的_deepseek_key>
PAIRWISE_EVAL_BASE_URL=https://api.deepseek.com
PAIRWISE_EVAL_MODEL=deepseek-chat
PAIRWISE_EVAL_TIMEOUT=300
```

## 3. 先做 dry-run（不访问API，不改metrics）

```powershell
python scripts/run_creval_vs_untrained_pipeline.py --dry-run --overwrite
```

预期：

- 只会生成 `*_creval.jsonl` 和 `*_llm_pairwise.jsonl` 的占位内容。
- 不会 merge 回 `*_metrics.jsonl`，也不会做分析。

## 4. 正式执行（访问API，生成完整对比）

```powershell
python scripts/run_creval_vs_untrained_pipeline.py --overwrite --allow-missing
```

这个命令会自动完成：

1. CrEval 打分（pairwise）
2. 未训练LLM baseline 打分（pairwise）
3. merge 两条外部分数字段到三份 Qwen metrics
4. 生成两套 scale regression 结果
5. 生成 Creval-vs-baseline 对照表
6. 生成 win-rate 汇总表

## 5. 输出文件（重点）

### 5.1 明细评分文件

- `data/generated/main_qwen3_4b_creval.jsonl`
- `data/generated/main_qwen3_8b_creval.jsonl`
- `data/generated/main_qwen3_14b_creval.jsonl`
- `data/generated/main_qwen3_4b_llm_pairwise.jsonl`
- `data/generated/main_qwen3_8b_llm_pairwise.jsonl`
- `data/generated/main_qwen3_14b_llm_pairwise.jsonl`

### 5.2 回归结果

- `results/tables/creval_scale_regression.csv`
- `results/tables/creval_auto_compare.csv`
- `results/tables/llm_pairwise_scale_regression.csv`
- `results/tables/llm_pairwise_auto_compare.csv`
- `results/tables/creval_vs_llm_pairwise_compare.csv`

### 5.3 Win-rate 结果

- `results/tables/pairwise_winrate_summary.csv`
- `results/tables/pairwise_winrate_by_genre.csv`
- `results/tables/creval_vs_llm_pairwise_winrate_compare.csv`

## 6. 只重算 win-rate（可选）

如果你已经完成 merge，只想重算 win-rate 表：

```powershell
python scripts/summarize_pairwise_winrate.py \
  --fields creval llm_pairwise \
  --inputs data/generated/main_qwen3_4b_metrics.jsonl data/generated/main_qwen3_8b_metrics.jsonl data/generated/main_qwen3_14b_metrics.jsonl
```

## 7. 如果你的输入不在默认路径

例如使用你自己目录下的 metrics 文件：

```powershell
python scripts/run_creval_vs_untrained_pipeline.py \
  --overwrite --allow-missing \
  --inputs F:/your_folder/main_qwen3_4b_metrics.jsonl F:/your_folder/main_qwen3_8b_metrics.jsonl F:/your_folder/main_qwen3_14b_metrics.jsonl
```

## 8. 论文描述建议（新增段落）

建议保持原文不替换，只新增说明：

- CrEval：训练过的 pairwise 创意评估器。
- baseline：普通未训练LLM pairwise 评估器（OpenRouter 或 DeepSeek API）。
- 二者采用同一对齐键、同一候选/参考对，评分映射为候选侧 `1/0.5/0`。
- 报告两类结果：
  - 与原自动指标同口径的二次项回归比较。
  - 直接 win-rate（含 tie）的对照比较。
