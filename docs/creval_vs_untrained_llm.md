# CrEval vs 普通未训练LLM 评估器对照流程

这份文档是新增流程，不替换现有 judge，也不改论文主流程。

目标：

1. 使用 CrEval 作为训练过的 pairwise creativity evaluator。
2. 并行使用普通未训练LLM作为 pairwise baseline evaluator。
3. 在同一批 Qwen 小模型结果上做并行比较，并导出对照表。

## 新增脚本

- scripts/score_existing_results_with_pairwise_llm.py
  - 对齐逻辑与 CrEval 脚本一致。
  - 输出 `*_llm_pairwise.jsonl`（可通过参数改后缀）。
- scripts/run_creval_vs_untrained_pipeline.py
  - 一键串联：Creval 打分 -> baseline 打分 -> merge -> 回归 -> 对照表。

## 环境变量

在项目 `.env` 中可选增加：

```env
PAIRWISE_EVAL_API_KEY=
PAIRWISE_EVAL_BASE_URL=
PAIRWISE_EVAL_MODEL=gpt-4o-mini
PAIRWISE_EVAL_TIMEOUT=300
```

说明：

- 若 `PAIRWISE_EVAL_API_KEY` 为空，会回退到 `OPENAI_API_KEY`。
- 若 `PAIRWISE_EVAL_BASE_URL` 为空，会回退到 `OPENAI_BASE_URL`，再回退到 `http://127.0.0.1:8000/v1`。

CrEval 端推荐（与你当前环境一致）使用 WSL 侧服务，Windows 侧只发 API 请求。

```env
CREVAL_API_KEY=0
CREVAL_BASE_URL=http://127.0.0.1:8000/v1
CREVAL_MODEL_NAME=gpt-3.5-turbo
CREVAL_API_TIMEOUT=1200
```

也兼容上游变量名：`CREVAL_API_BASE_URL`、`CREVAL_API_MODEL`。

### baseline 走 OpenRouter（未训练LLM）

```env
PAIRWISE_EVAL_API_KEY=<your_openrouter_key>
PAIRWISE_EVAL_BASE_URL=https://openrouter.ai/api/v1
PAIRWISE_EVAL_MODEL=openai/gpt-4o-mini
PAIRWISE_EVAL_TIMEOUT=300
```

### baseline 走 DeepSeek（未训练LLM）

```env
PAIRWISE_EVAL_API_KEY=<your_deepseek_key>
PAIRWISE_EVAL_BASE_URL=https://api.deepseek.com
PAIRWISE_EVAL_MODEL=deepseek-chat
PAIRWISE_EVAL_TIMEOUT=300
```

说明：baseline 脚本会自动回退读取 `DEEPSEEK_API_KEY` / `DEEPSEEK_BASE_URL`。

## WSL 启动 CrEval 服务（建议）

在 WSL 中必须启动 **API server**（`inference.py` 是交互客户端，不是服务端）。

```bash
cd /mnt/f/2026spInnovation/CrEval
export CREVAL_API_BASE_URL=http://127.0.0.1:8000/v1
export CREVAL_API_MODEL=gpt-3.5-turbo
export CREVAL_API_TIMEOUT=1200
export API_VERBOSE=0

# 方式A：LLaMA-Factory API（推荐）
API_PORT=8000 CUDA_VISIBLE_DEVICES=0 llamafactory-cli api creval_api.yaml
```

如需手工验证，再另开终端运行 `inference.py` 做交互测试，但它本身不提供 HTTP 服务。

核心要求是：
`http://127.0.0.1:8000/v1` 在 Windows 侧可访问。

可在 Windows PowerShell 检查：

```powershell
Test-NetConnection -ComputerName 127.0.0.1 -Port 8000
python -c "import httpx; print(httpx.get('http://127.0.0.1:8000/v1/models', timeout=3.0).status_code)"
```

`API_VERBOSE=0` 时只保留访问日志（例如 `INFO: 127.0.0.1 ... 200 OK`），不会打印整段 request JSON。

## 一键执行

在仓库根目录运行：

```powershell
python scripts/run_creval_vs_untrained_pipeline.py --overwrite --allow-missing
```

CrEval 偶发 502 已在脚本内做自动重试（默认 5 次）。

如需更强重试：

```powershell
python scripts/run_creval_vs_untrained_pipeline.py --overwrite --allow-missing --creval-retries 8 --creval-sleep-seconds 2.0
```

若 baseline 模型出现 `403 This model is not available in your region`，可在命令中加 fallback：

```powershell
python scripts/run_creval_vs_untrained_pipeline.py --overwrite --allow-missing --creval-retries 8 --creval-sleep-seconds 2.0 --baseline-fallback-models deepseek/deepseek-chat-v3-0324 deepseek-chat
```

如果你只想先检查对齐，不访问API：

```powershell
python scripts/run_creval_vs_untrained_pipeline.py --dry-run --overwrite
```

说明：`--dry-run` 会自动跳过 merge 和 analysis，保证不改动现有 `*_metrics.jsonl`。

## 502/中断时的恢复执行

1. 只重跑 CrEval 打分（带重试，避免全流程反复）：

```powershell
python scripts/score_existing_results_with_creval.py --reference-model gen_openai_4o --overwrite --retries 8 --sleep-seconds 2.0
```

1. 若 `*_creval.jsonl` 已可用，跳过 CrEval，仅继续 baseline + merge + 分析：

```powershell
python scripts/run_creval_vs_untrained_pipeline.py --skip-creval --overwrite --allow-missing
```

## 产物说明

除原有产物外，会新增：

- `data/generated/main_qwen3_4b_llm_pairwise.jsonl`
- `data/generated/main_qwen3_8b_llm_pairwise.jsonl`
- `data/generated/main_qwen3_14b_llm_pairwise.jsonl`

并在 merge 后于 `*_metrics.jsonl` 新增字段：

- `creval`
- `llm_pairwise`

回归与对照表：

- `results/tables/creval_scale_regression.csv`
- `results/tables/creval_auto_compare.csv`
- `results/tables/llm_pairwise_scale_regression.csv`
- `results/tables/llm_pairwise_auto_compare.csv`
- `results/tables/creval_vs_llm_pairwise_compare.csv`

图像：

- `results/figures/scale_inverted_u_creval.png`
- `results/figures/scale_inverted_u_llm_pairwise.png`

## 解释建议

论文中建议作为新增外部验证通道描述：

- CrEval 表示训练过的专用创意评估器。
- llm_pairwise 表示普通未训练LLM评估器 baseline。
- 二者都以 pairwise 协议输出候选侧分数（1/0.5/0）。
- 原有自动指标与原 judge 结果保持不变。
