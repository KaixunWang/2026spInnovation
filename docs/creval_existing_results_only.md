# 用现有结果补 CrEval 外部评分

这份文档只覆盖一个目标：

不重跑生成，不重跑 metrics，不训练模型，只使用仓库已有结果，为 Qwen 小模型补一条 CrEval 外部评分。

## 你会得到什么

跑完后会得到：

- data/generated/main_qwen3_4b_creval.jsonl
- data/generated/main_qwen3_8b_creval.jsonl
- data/generated/main_qwen3_14b_creval.jsonl

合并后会更新：

- data/generated/main_qwen3_4b_metrics.jsonl
- data/generated/main_qwen3_8b_metrics.jsonl
- data/generated/main_qwen3_14b_metrics.jsonl

最后会产出：

- results/tables/creval_scale_regression.csv
- results/tables/creval_auto_compare.csv
- results/figures/scale_inverted_u_creval.png

## 0. 前提

当前仓库里已经有这些文件：

- data/generated/main_metrics.jsonl
- data/generated/main_qwen3_4b_metrics.jsonl
- data/generated/main_qwen3_8b_metrics.jsonl
- data/generated/main_qwen3_14b_metrics.jsonl

这些文件已经足够，不需要再生成新 rewrite。

## 1. 单独准备 CrEval 环境

建议在仓库外单独准备，不要和当前实验环境混在一起。

```powershell
git clone https://github.com/Aman-4-Real/CrEval.git
cd CrEval
conda create -n creval python=3.10
conda activate creval
pip install -r requirements.txt
```

如果后面 llamafactory-cli 不存在，再补装与 README 对齐的 LLaMA-Factory 版本。

## 2. 下载 CrEval 模型

建议优先用 CrEval-7b。

你需要两部分：

- base model: Qwen2.5-7B-Instruct
- adapter: Aman/CrEval-7b

然后修改上游仓库的 creval_api.yaml：

```yaml
model_name_or_path: D:/models/Qwen2.5-7B-Instruct
adapter_name_or_path: D:/models/CrEval-7b
template: qwen
finetuning_type: lora
```

## 3. 启动 CrEval API

你当前环境推荐在 WSL 中启动（你已经验证 Windows 本地直接跑会更慢）。

在 WSL 的 CrEval 目录：

```bash
cd /mnt/f/2026spInnovation/CrEval
export CREVAL_API_BASE_URL=http://127.0.0.1:8000/v1
export CREVAL_API_MODEL=gpt-3.5-turbo
export CREVAL_API_TIMEOUT=1200
export API_VERBOSE=0

# 启动 API server（推荐）
API_PORT=8000 CUDA_VISIBLE_DEVICES=0 llamafactory-cli api creval_api.yaml
```

服务起来之后，保持这个终端不要关。

说明：`inference.py` 是交互客户端，不是 API server。批量评分由主仓库脚本自动调用 HTTP API 完成，不需要手工逐条输入。
`API_VERBOSE=0` 可以关闭请求体大段日志，只保留一行访问日志。

## 4. 可选：先做一次官方交互测试

在 CrEval 仓库目录新开一个终端：

```powershell
conda activate creval
python inference.py
```

随便输入一个 query 和两个 response，确认服务正常返回。

也可在 Windows 侧做连通性检查：

```powershell
Test-NetConnection -ComputerName 127.0.0.1 -Port 8000
python -c "import httpx; print(httpx.get('http://127.0.0.1:8000/v1/models', timeout=3.0).status_code)"
```

## 5. 回到当前项目，填写 .env

当前项目根目录的 .env 至少加上这三行：

```env
CREVAL_API_KEY=0
CREVAL_BASE_URL=http://127.0.0.1:8000/v1
CREVAL_MODEL_NAME=gpt-3.5-turbo
CREVAL_API_TIMEOUT=1200
```

说明：

- 新脚本也兼容上游变量名 `CREVAL_API_BASE_URL` 与 `CREVAL_API_MODEL`
- 如果你的服务端口不是 8000，只改 CREVAL_BASE_URL

## 6. 先做一次 dry-run 检查配对

在当前项目根目录：

```powershell
python scripts/score_existing_results_with_creval.py --inputs data/generated/main_qwen3_4b_metrics.jsonl --limit 1 --overwrite --dry-run
```

你应该看到类似：

- sample key
- sample query chars
- 写出 main_qwen3_4b_creval.jsonl

这是为了确认：

- 当前仓库的 Qwen 结果和 GPT 参考能按 T3 精确对齐
- CrEval 批量脚本正常工作

如果只是 dry-run 检查，记得把这份临时文件删掉：

```powershell
Remove-Item data/generated/main_qwen3_4b_creval.jsonl
```

## 7. 正式跑 CrEval 批量评分

```powershell
python scripts/score_existing_results_with_creval.py --reference-model gen_openai_4o --overwrite --retries 8 --sleep-seconds 2.0
```

默认会对三份 Qwen 小模型 metrics 文件执行评分，并写出三份 *_creval.jsonl。

这里做的是：

- 候选：Qwen 小模型的 T3 rewrite
- 参考：main_metrics.jsonl 中对齐的 gen_openai_4o T3 rewrite
- 打分方式：CrEval pairwise 判谁更有创意

分数定义：

- 1.0：Qwen 胜
- 0.5：平
- 0.0：Qwen 负

如果遇到偶发 502：

- 脚本会先自动重试；超过重试后会把该条写成 `reason=request_error`，不会整批中断。
- 可重复执行同一命令（`--overwrite`）进行补跑。

## 8. 合并到现有 metrics

```powershell
python scripts/merge_external_eval_into_metrics.py data/generated/main_qwen3_4b_metrics.jsonl data/generated/main_qwen3_4b_creval.jsonl --field-name creval --allow-missing
python scripts/merge_external_eval_into_metrics.py data/generated/main_qwen3_8b_metrics.jsonl data/generated/main_qwen3_8b_creval.jsonl --field-name creval --allow-missing
python scripts/merge_external_eval_into_metrics.py data/generated/main_qwen3_14b_metrics.jsonl data/generated/main_qwen3_14b_creval.jsonl --field-name creval --allow-missing
```

注意：

- 这里只合并 Qwen 三份文件
- 不会覆盖现有 judge 字段
- 只是新增 creval 字段

## 9. 生成 CrEval 对比结果

```powershell
python scripts/external_eval_scale_regression.py --field-name creval --score-key score --inputs data/generated/main_qwen3_4b_metrics.jsonl data/generated/main_qwen3_8b_metrics.jsonl data/generated/main_qwen3_14b_metrics.jsonl
```

输出文件：

- results/tables/creval_scale_regression.csv
- results/tables/creval_auto_compare.csv
- results/figures/scale_inverted_u_creval.png

你重点看第二个：

- results/tables/creval_auto_compare.csv

里面会直接给出：

- 每个 Qwen 模型的 mean_creval_score_T3
- 与 creativity_auto 的二次项比较

## 10. 如果你要写进论文

建议只新增一小节，不改原有章节。

建议表达为：

- CrEval 是新增的外部 pairwise creativity evaluator
- 由于它是 pairwise 协议，当前报告的是 “Qwen 相对 matched GPT reference 的胜率型分数”
- 原有 C_auto 与原 judge 结果保持不变

## 最短命令清单

如果你已经把 CrEval 服务开起来，当前项目里真正要敲的就是下面这些命令：

```powershell
python scripts/score_existing_results_with_creval.py --reference-model gen_openai_4o --overwrite
python scripts/merge_external_eval_into_metrics.py data/generated/main_qwen3_4b_metrics.jsonl data/generated/main_qwen3_4b_creval.jsonl --field-name creval --allow-missing
python scripts/merge_external_eval_into_metrics.py data/generated/main_qwen3_8b_metrics.jsonl data/generated/main_qwen3_8b_creval.jsonl --field-name creval --allow-missing
python scripts/merge_external_eval_into_metrics.py data/generated/main_qwen3_14b_metrics.jsonl data/generated/main_qwen3_14b_creval.jsonl --field-name creval --allow-missing
python scripts/external_eval_scale_regression.py --field-name creval --score-key score --inputs data/generated/main_qwen3_4b_metrics.jsonl data/generated/main_qwen3_8b_metrics.jsonl data/generated/main_qwen3_14b_metrics.jsonl
```
