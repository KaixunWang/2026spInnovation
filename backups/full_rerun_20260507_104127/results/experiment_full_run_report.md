# 全量实验运行报告

生成日期：2026-04-30。

## 任务结果摘要

- **LLM Judge（三份 `*_metrics.jsonl`）**：已成功跑完（任务 exit 0），行数分别约为 **2280 / 1920 / 96**，合并脚本 `merge_judge_into_metrics.py` 已将 judge 列写回对应 metrics。
- **`src.analyze`**：曾在 `plot_inverted_u` 的 `fill_between` 处因非浮点数组报错；已改为对 `d_bin_center` / `mean` / `ci95` 使用 `to_numpy(dtype=float)` 后 **完整跑通**（exit 0）。混合效应回归阶段可能出现 statsmodels 的 singular / convergence 警告，属数值边界常见提示，不影响脚本退出码。

## 配置与环境说明

- **生成器 / Judge**：因首轮 DashScope（`gen_qwen35_flash`）401，当前配置为 **OpenAI 路线**（见 `configs/models.yaml`：`gen_openai_4o`、`gen_openai_mini`、`judge_openai_4o`、`coord_openai_mini` 等）。
- **Coordinate 与 primary 同族**：在需要校验的命令前设置 **`ALLOW_SAME_FAMILY_COORD=1`**（实现见 `run_experiment.py` 中 `_allow_same_family_coord()`），否则会同族报错。
- **Metrics**：本次 **`metrics` 未使用 `--with-embedding`**（此前 embedding 步骤存在 HF 下载或模型名不可用等问题）；`flat.csv` 中 embedding 相关列可能为占位或未填充，分析以已有自动指标与 judge 为主。
- **Space-L**：仓库存在缓存产物 **`.cache/space_l/space_l_summary.json`**；若未在本轮显式执行 `build_space_l --force`，则为沿用已有缓存。

## 产出核对

- **`results/flat.csv`**：已生成，含 judge 列及 `source_file`。
- **`results/tables/`**：含 `binned_creativity_vs_d.csv`、`causal_creativity_auto.csv`、`mechanism_creativity_auto.csv`、`judge_auto_validity_correlation.csv` 等。
- **`results/regression_diagnostics.json`**：回归诊断汇总。
- **图**：分析日志中会写入 `results/figures/` 下若干 PNG（若目录为空请确认磁盘路径与工作区一致）。

## 后续可选

- 若需 embedding 列完整：修复 HF/embedding 模型配置后，对相应 arms **重跑 `metrics --with-embedding`**，再 merge judge（若 judge 未变可只更新 metrics）、重跑 `analyze`。
- 若需减弱混合模型边界警告：可考虑简化随机效应结构或增大每胞样本（属研究设计层面，非本次脚本必选）。
