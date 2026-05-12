# 测量模型与指标角色（审稿对照）

本文档固定**操作化定义**，避免把自动指标误读成「人类意义上的价值判断」。

## 主因变量（主结论）

- **`creativity_auto`（主因变量）**  
  `creativity_auto = novelty_auto_combined × value_auto`（见 `src/run_experiment.py` 中 `cmd_metrics` 写入的字段）。

## 自动指标块（`metrics`）

### Novelty（`novelty`）

- 由三项经文体基线归一后的量合成 **`novelty.auto_combined`**（见 `src/metrics/novelty.py`）：
  - `distinct2_rel`：与**同文体源文库**相比的 distinct-2 相对化；
  - `novel_ngram_rel`：与同文体源文对的 n-gram overlap 基线相比；
  - `embedding_distance_rel`：与同文体源文 embedding 距离基线相比（若 `--with-embedding`，embedding 由 `roles.embedder` 决定，可为本地 SBERT 或 OpenAI API）。

### Value（`value_auto`）

- **`fidelity_proxy`**：默认取 **NLI entailment**（`facebook/bart-large-mnli`，premise=源文，hypothesis=生成）；若 NLI 缺失则**退回**用 `coherence_auto` 代替（见 `cmd_metrics`）。
- **`coherence_auto`**：由 **GPT-2 困惑度** `perplexity(gen)` 经 `perplexity_to_unit` 映射到 \([0,1]\)（不是 LLM judge）。
- **`value_auto`**：`combine_value_arith(fidelity_proxy, coherence_auto)`，即 **NLI 代理与流畅度代理的算术平均**。  
  这是**操作化命名**；不等同于人类评审的「价值」或「忠实度」本体。

### 与文体的关系

- Novelty 三项对 **genre 内源文分布**做了归一；**`value_auto` 未做同类 genre-relative 归一**，因此回归中可能出现 **genre 主效应或交互**，需在分析表中单独分层报告（`genre_t3_d_value_stratified.csv`）并在正文解释来源。

## LLM Judge（`judge`）

- **角色定位：收敛效度 / 敏感性证据**，用于检查自动指标是否与人类式评分同向相关；**不替代** `creativity_auto` 作为主因变量，除非在独立研究中事先声明并改主分析计划。
- 产出 `*_judged.jsonl`；与 metrics 合并后，`analyze` 会输出 **`judge_auto_validity_correlation.csv`** 与 **`judge_auto_validity_matrix.png`**（Pearson 相关矩阵），供附录引用。

## H1 预注册（`configs/experiment.yaml` → `main.h1_prereg`）

- 顶点 \(d\) 的**分位区间**、**证伪规则**（二次项 \(p\) 与拟合顶点相对样本 \(d\) 分位）以 YAML 为准；`analyze` 写入 **`h1_preregistration_check.csv`**，并在回归前生成 **`t3_d_H_distribution_main.png`** 检查 \(d_H\) 覆盖。

## 坐标诊断（`coord_reliability`）

- 默认 **`coord_scoring.backend: hf`**：源 \((S,R)\) 为确定性 HF 坐标（见 `docs/experiment_workflow.md` 第 1a 步）。在跑 **`main`** 之前执行 **`python -m src.run_experiment coord_reliability`**，读取预计算坐标，输出 **`results/tables/coord_reliability.csv`**（genre 均值/质心距离/ANOVA 等），作为 \(d_H\) 推断链的前置材料。
- 若使用 **`backend: llm`**，可沿用历史上「双 coordinate 模型 + ICC」叙事；与当前默认 HF 管线二选一即可。
