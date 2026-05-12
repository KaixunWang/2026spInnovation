# 全量实验运行报告

生成日期：**2026-05-09**（本轮：`judge` 补全 → `merge_judge_into_metrics` → `python -m src.analyze`）。

## 任务结果摘要

- **LLM Judge**：四份输入均已具备完整 `*_judged.jsonl`（`main` / `main_continuous` / `mechanism` / `multihop`），与各自 `*.jsonl` 行数一致；`merge_judge_into_metrics.py` 已将 `judge` 字段写回对应 `*_metrics.jsonl`。
- **`src.analyze`**：本轮 **`python -m src.analyze`** 已完整跑通（exit 0）。混合效应回归阶段可能出现 statsmodels 的 singular / convergence / Hessian 类警告，属数值边界常见提示，不影响脚本退出码。

## 数据核对（`results/flat.csv`）

- **总行数**：4716。
- **按 `source_file` 计数**：`main_metrics` 2280，`main_continuous_metrics` 2280，`mechanism_metrics` 60（top-15 高冲突 cell × M0–M3，`n_repeat=1`），`multihop_metrics` 96。

## 配置与环境说明

- **生成器 / Judge**：以仓库当前 `configs/models.yaml` 与 `.env` 为准；此前若遇代理 TLS 超时，可通过稳定网络后 **`judge --resume`** 续跑（已在本轮前用于补全断点）。
- **Coordinate 与 primary 同族**：若命令报错，可在需要时设置 **`ALLOW_SAME_FAMILY_COORD=1`**（见 `run_experiment.py` 中 `_allow_same_family_coord()`）。
- **Metrics**：若某次 `metrics` 未使用 `--with-embedding`，`flat.csv` 中 embedding 相关列可能为占位；分析以已有自动指标与 judge 为主。
- **Space-L**：若未显式执行 `build_space_l --force`，可能沿用 **`.cache/space_l/`** 下已有缓存。

## 产出路径（本轮已刷新）

- **`results/flat.csv`**：合并后的宽表。
- **`results/tables/`**：如 `binned_creativity_vs_d.csv`、`causal_creativity_auto.csv`、`mechanism_creativity_auto.csv`、`h1_preregistration_check.csv`、`judge_auto_validity_correlation.csv`、`multihop_prediction_check.csv`、`novelty_collinearity.csv` 等。
- **`results/regression_diagnostics.json`**：各 metric 的主回归与方向回归诊断。
- **`results/figures/`**：如 `t3_d_H_distribution_main.png`、`inverted_u_creativity_auto.png`、`directional_field_creativity_auto.png`、`pair_heatmap_creativity_auto.png`、`multihop_trajectories.png`、`judge_auto_validity_matrix.png` 等（时间戳以本机文件为准）。

## 后续可选

- 若需 embedding 列完整：修复 HF/embedding 配置后对相应 arms **重跑 `metrics --with-embedding`**，必要时再 merge judge、重跑 `analyze`。
- 若需减弱混合模型边界警告：可考虑简化随机效应结构或调整样本设计（属研究设计层面，非脚本必选）。
