# 当前实验计划流程（与 v2 研究方案对齐）

本文档描述**在本仓库代码中**如何按阶段跑完整研究管线，并对应说明方案中的**理论设计、因果结构、指标与产出**。不替代 `docs/proposal.md` 的学术叙述，侧重「先做什么、再做什么、为什么」。

**测量与效度链**（自动指标 vs judge、H1 预注册、坐标信度）见 **`docs/rubrics.md`**。

---

## 总览：一条主线 + 三条支线


| 主线    | 内容                                    | 对应方案章节     |
| ----- | ------------------------------------- | ---------- |
| 语料与坐标 | 60 篇英文原文（4 文体 × 15）；原文在 Space-H 上打分；可选构建 Space-L | §5、§3.2、§4 |
| 主实验   | T0–T3 四臂生成                            | §2、§10     |
| 评估    | 自动指标 + 可选 LLM judge                   | §6、§7      |
| 支线 A  | 机制实验 M1/M2/M3                         | §8、H6      |
| 支线 B  | 多轮杂交                                  | §9         |
| 支线 C  | 分析与作图                                 | §10、H1–H7  |


命令入口统一为：`python -m src.run_experiment <子命令>`；统计与图表：`python -m src.analyze`。

长耗时子命令默认在 **stderr** 用 **tqdm** 显示进度；日志或 CI 中若不想看到进度条，可设环境变量 **`NO_TQDM=1`**。

---

## 第 0 步：环境与语料（任何实验前必做）

### 你要做什么

1. **依赖**：在项目根目录执行 `pip install -e .`（或按 `README.md` 安装说明）。
2. **API**：复制 `.env.example` 为 `.env`，填入至少一个可用的生成接口；若跑 judge，建议第二个厂商或第二个模型角色（见 `configs/models.yaml` 的 `roles`）。
3. **语料（推荐半自动）**：
   - 先运行 `python -m scripts.build_source_corpus --target-per-genre 15` 自动采集：
     - `academic <- gfissore/arxiv-abstracts-2021`
     - `narrative <- sanps/GutenbergFiction`
     - `poetry <- merve/poetry`
   - `essay` 保持手工（Wikisource/public-domain essays，见 `data/source_texts/essay/MANUAL_COLLECTION.md`）。
   - 当前仓库目标为 **4 文体 × 15 = 60 篇**（每篇带 YAML front-matter，见 `data/source_texts/README.md`）。

### 为什么（对应 plan）

- 方案限定为 **英文、文学性短文（约 150–300 词）**，便于控制长度与评估成本（§5、§13）。
- **双空间**与 **judge** 需要稳定可复现的运行环境与密钥管理。

### 自检命令

```bash
python -m scripts.smoke_offline          # 不联网：本地模块与数学逻辑
python -m scripts.build_source_corpus --target-per-genre 15   # 半自动采集语料（三类自动+essay手工）
python -m src.run_experiment validate_corpus --expected-per-genre 15   # 校验 60 篇语料
```

---

## 第 1 步：原文在 Space-H 上坐标打分（`coord`）

### 命令

```bash
python -m src.run_experiment coord
```

（可选 `--recompute` 强制重算。）

### 做什么

- 用配置里的 **coordinate_scorer** 模型，按 `configs/personalities.yaml` 里的量表，对每个 **source** 打出认知轴 `S` 与修辞风险轴 `R`，归一化到 `[-1,1]`，得到 **Space-H 中的 `src_vec`**。
- 结果写入 `data/generated/coord_scores.jsonl`。
- 同步导出 `results/tables/coord_space_h_distribution.csv` 与 `results/figures/coord_space_h_distribution.png`，用于检查 source 在 S-R 平面覆盖是否均匀。

### 为什么（对应 plan）

- **Space-H** 是理论可解释平面（Dual Process × Register）（§0.1–0.2、§3.1）。
- 后续 **冲突强度 `d_H`** 与 **方向向量 `Δ_H = tgt − src`** 都依赖 `src_vec`（§4）。

---

## 第 1a 步：确定性 Space-H 源坐标（推荐，`coord_scoring.backend: hf`）

默认 **`configs/experiment.yaml`** 中 **`coord_scoring.backend`** 为 **`hf`**：源文 \((S,R)\) 由 HuggingFace 分类模型 + StyleDistance 锚轴计算，**同一文本结果可复现**，不再依赖 LLM 坐标打分。

### 一次性离线管线

```bash
python scripts/generate_anchors.py          # LLM：写出 data/anchors.json（四极 × 8 篇）
python scripts/build_axis_vectors.py        # StyleDistance：data/axis_vectors.npz
python scripts/recompute_coords.py          # 全语料：data/source_coords.jsonl
python scripts/validate_coords.py           # 按 genre 验收 S/R 方向（可选阈值）
```

- **`coord` / `main`**：对每个源文优先读 **`data/source_coords.jsonl`**；缺行则现场用同一套 HF 公式补算。
- 产物仍写入 **`data/generated/coord_scores.jsonl`**（字段 `S_mean` / `R_mean` 等），下游 **`d_H`** 逻辑不变。
- 若需恢复旧行为：将 **`coord_scoring.backend`** 设为 **`llm`**（仍使用 `coordinate_scorer` + `personalities` 量表 prompt）。

---

## 第 1b 步：坐标诊断（`coord_reliability`，**须在 `main` 之前**）

### 命令

```bash
python -m src.run_experiment coord_reliability
```

（可选 `--limit-per-genre N` 做小规模试跑。）

### 做什么（`backend: hf`）

- 读取 **`coord_scoring.source_coords_path`**（默认 **`data/source_coords.jsonl`**；若缺失则回退 **`data/generated/coord_scores.jsonl`**）。
- 按 **genre** 汇总 **S / R** 的均值与标准差；计算 **genre 质心之间的欧氏距离**；并对 S、R 分别做 **单因素 ANOVA** 与 **Kruskal–Wallis**（跨 genre），写入 **`results/tables/coord_reliability.csv`**（含逐源行 + 摘要行）。
- **不会**覆盖 `coord_scores.jsonl`。

### 做什么（`backend: llm`，遗留）

- 可使用旧版双 LLM 打分 + ICC 流程（见历史提交）；当前默认配置以 HF 坐标为主，**`coord_reliability.scorer_models` 在 `hf` 模式下不参与**。

### 为什么

- 确定性坐标下 **双 LLM 的 ICC 不再适用**；改为报告 **genre 间几何分离**，作为 \(d_H\) 推断链的前置材料。

---

## 第 2 步（可选但推荐）：构建 Space-L（`build_space_l`）

### 命令

```bash
python -m src.run_experiment build_space_l
```

（`--force` 可强制重建缓存。）

### 做什么

- 对每个 persona，在中性种子话题上生成大量 reference 短文；用 SBERT 聚合为 `v_P`，再 PCA 得到 **Space-L**；可与 Space-H 做 Procrustes 对齐摘要（见 `.cache/space_l/space_l_summary.json`）。

### 为什么（对应 plan）

- 回应「轴是否任意」：**learned 空间**与手工空间互证（§3.2、**H7**）。
- 注意：API 与本地 embedding 成本较高；方案建议可先子集或调低 `configs/experiment.yaml` 里 `space_l.n_ref_per_persona` 做 pilot。

---

## 第 3 步：主实验 — 四臂生成（`main`）

### 命令

```bash
python -m src.run_experiment main --overwrite
```

（`--limit-per-genre N` 可用于小规模试跑。）

### 做什么

对每篇原文、每个配置的 **generator**、重复 `n_repeat` 次，生成：


| 条件     | 含义                                         | 因果角色（§2）           |
| ------ | ------------------------------------------ | ------------------ |
| **T0** | Identity，原文拷贝                              | 测量噪声基线             |
| **T1** | Same-personality，与 `src_vec` 最近 persona 改写 | **纯改写**，近零冲突       |
| **T2** | Random-personality，随机 persona 改写           | **Placebo**：任意风格变化 |
| **T3** | Cross-personality，指定目标 persona 改写          | **Treatment**：有向冲突 |


输出：`data/generated/main.jsonl`（每行一条记录，含 `condition`、`d_H`、`delta_H`、`text` 等）。

### 为什么（对应 plan）

- **准因果**：后续分析用 **Δ_creativity = T3 − T1**、**placebo gap = T3 − T2** 区分「冲突驱动」与「随便改写」（§2、§10）。
- **H1 倒 U**、**H3 方向**、**H4 文体** 都在此数据上拟合。

---

## 第 4 步：自动指标（`metrics`）

### 命令

单臂：

```bash
python -m src.run_experiment metrics --inputs data/generated/main.jsonl
```

全量常见写法（在 `main` / `mechanism` / `multihop` 都生成完之后；`--with-embedding` 可选）：

```bash
python -m src.run_experiment metrics --with-embedding --inputs data/generated/main.jsonl data/generated/mechanism.jsonl data/generated/multihop.jsonl
```

（可加 `--skip-nli` / `--skip-ppl` 等减轻负载。）

### 做什么

- **Novelty**：distinct-2、novel n-gram、可选 embedding 距离；并对 **文体基线归一化**（§6.1），避免诗歌/学术不可比。
- **Value 主定义**：`Value_arith = (fidelity_proxy + coherence_auto)/2`（正式跑主结论使用）。
- **敏感性定义**：并行输出 `Value_geom = sqrt(fidelity_proxy * coherence_auto)` 与 `Creativity_auto_geom`，用于检验 H1 结论稳健性。
- 辅助：情感偏移、结构 Kendall τ、归一化编辑距离等（§6.5）。

### 为什么（对应 plan）

- 把「变化」从 **Novelty × Value** 框架上收紧为可发表的复合指标（§0.3、§6）。

输出：例如 `data/generated/main_metrics.jsonl`。

---

## 第 5 步（可选）：LLM-as-Judge（`judge`）

### 命令

```bash
python -m src.run_experiment judge
```

默认会处理 `main` / `mechanism` / `multihop` 下已存在的 jsonl；也可用 `--inputs` 指定文件。

### 做什么

- 双 judge（若配置了两个）对 **Novelty / Surprise / Imagery / Fidelity / …** 绝对打分；可与自动通道融合（见 `docs/rubrics.md`）。

### 为什么（对应 plan）

- 缓解 **compression in the middle**；与生成器不同家族可降低自吹偏差（§6.6）。

输出：`*_judged.jsonl`。

---

## 第 6 步（可选）：机制实验（`mechanism`）

### 命令

```bash
python -m src.run_experiment mechanism --overwrite
```

### 做什么

在 **高冲突（高 `d_H`）** 子集上，对同一 (source, target) 比较：

- **M0**：空 checklist 长度控制（token 结构对照）  
- **M1**：联合 prompt 一次生成  
- **M2**：先忠实复述，再套 persona  
- **M3**：联合 + 显式命题 checklist

### 为什么（对应 plan）

- 检验 **H6**：高冲突下 Value 坍塌是否来自「风格阶段覆盖语义阶段」（§8）。

---

## 第 7 步（可选）：多轮杂交（`multihop`）

### 命令

```bash
python -m src.run_experiment multihop --overwrite
```

（`--fast` 可跳过对中间稿的坐标重估以省 API。）

### 做什么

- 按 `configs/experiment.yaml` 里预设的 persona 序列，做 `source → P_A → P_B → …`，记录轨迹。
- 按可证伪预测评估：`high-conflict bridge` 路径终点创意分是否高于匹配 d-bin 的直接 T3（输出 `results/tables/multihop_prediction_check.csv`）。

### 为什么（对应 plan）

- **漂移 / attractor / 路径依赖**（§9），把静态 pairwise 扩成动力系统视角。

输出：`data/generated/multihop.jsonl`。

**外推范围**：当前仅 `experiment.yaml` 中**预注册的两条 persona 路径**；更宽的路径族结论需扩展路径采样（见同文件 `multihop.scope_note`）。

---

## 第 8 步：信息论风格距离（与代码衔接）

### 说明

方案 §7 要求 **JSD(P_source, P_target)** 等；实现见 `src/info_theory.py`。若要把 JSD 并入每条生成的回归表，可在后续版本把 reference 分布与 `d` 一并写入 jsonl；当前 `analyze` 以 **d、d²、Δ、条件、文体** 为主回归。

### 为什么（对应 plan）

- **H5**：Creativity 与风格分布偏离是否也呈倒 U（channel capacity 叙事）（§0.5、§7）。

---

## 第 9 步：统计分析与出图（`analyze`）

### 命令

```bash
python -m src.analyze --metric creativity_auto
```

### 做什么

- 读入 `*_metrics.jsonl`（及可选 `*_judged.jsonl`），展平为 `results/flat.csv`。
- **H1 预注册诊断（先于主回归作图）**：`T3` 且 `main_metrics` 子集上 `d_H` 直方图 → `results/figures/t3_d_H_distribution_main.png`；并写 `results/tables/h1_preregistration_check.csv`（与 `configs/experiment.yaml` 中 `main.h1_prereg` 对照）。
- **Judge–自动指标效度**：`results/tables/judge_auto_validity_correlation.csv` 与 `results/figures/judge_auto_validity_matrix.png`（需已 merge judge）。
- **文体分层（T3 / main）**：`results/tables/genre_t3_d_value_stratified.csv`（检查 `value_auto` 是否被 genre 主效应驱动）。
- **主回归**：`metric ~ d + d² + genre + condition`（及混合效应近似）（**H1、H4**）。
- **方向回归**：`metric ~ ΔS + ΔR + …`（**H3**）。
- **拆解回归**：`Novelty_norm` 与 `Value` 分别回归，避免仅凭合成分解读倒 U。
- **共线诊断**：输出 `results/tables/novelty_collinearity.csv`（distinct/novel_ngram/embdist 的相关）。
- **winsorize 诊断**：输出 winsorized novelty/creativity 版本，替代单一路径硬 clip 结论。
- **因果对比表**：按 `d` 分桶的 T3−T1、T3−T2（§10）。
- **机制表**：高 d 下 M1 vs M2/M3（**H6**）。
- 图：倒 U 散点+二次拟合、方向场、多轮轨迹、热力图等 → `results/figures/`。

### 为什么（对应 plan）

- 把现象接到 **可证伪** 的系数与置信区间上，并服务论文图表清单（§10）。

---

## 第 10 步：Smoke（上线前小样本）

### 命令

```bash
python -m src.run_experiment smoke
```

### 做什么

- 极少原文、极少条件，验证 **coord + 生成 + 本地 novelty** 链路。

### 为什么

- 方案里程碑 W1 强调先 smoke 再全量（plan §12），避免大规模 API 浪费。

---

## 建议执行顺序（简表）

下文 **第 4–7 步**按「指标 / judge / 机制 / 多跳」分块说明各子命令做什么，**不表示**推荐执行先后。  
**全量跑通**时请按下表顺序：**先生成各臂原文（main → mechanism → multihop）**，再统一 **`metrics` → `judge` → `merge_judge_into_metrics`**，最后 **`analyze`**（否则支线未进 `*_metrics.jsonl` 就要补跑 metrics，或 judge 效度表不全）。

| 顺序 | 命令 | 依赖 |
| --- | --- | --- |
| 0 | `python -m scripts.smoke_offline` / `validate_corpus --expected-per-genre 15` | 无 / 语料 |
| 0b | `smoke`（可选） | .env + API |
| 1 | `coord_reliability` | `source_coords.jsonl` 或已跑的 `coord_scores.jsonl` |
| 2 | `coord` | API |
| 3 | `main --overwrite` | API；需已有 coord |
| 4 | `mechanism --overwrite` | API |
| 5 | `multihop --overwrite` | API |
| 6 | `build_space_l`（可选；成本高，可与 3–5 并行规划但通常放生成后） | API + 本地 SBERT |
| 7 | `metrics`（建议 `--inputs` 含 main / mechanism / multihop；可选 `--with-embedding`） | 各臂 `*.jsonl` |
| 8 | `judge --inputs …`（与上一步同一批 jsonl） | API |
| 9 | `python scripts/merge_judge_into_metrics.py <*_metrics.jsonl> <*_judged.jsonl> -o <*_metrics.jsonl>` | 上两步产物 |
| 10 | `python -m src.analyze --metric creativity_auto` | merge 后的 `*_metrics.jsonl` |


---

## 与假设（H1–H7）的对应速查


| 假设             | 主要数据与步骤                                        |
| -------------- | ---------------------------------------------- |
| **H1** 倒 U     | `main` + `metrics` → `analyze`：`t3_d_H_distribution_main.png` + `h1_preregistration_check.csv` + d、d² 系数 |
| **H2** 坍塌阈值    | 同上，对 Value 分段或 changepoint（可在 analyze 上扩展）     |
| **H3** 方向梯度    | `delta_H` + 方向回归 / 方向场图                        |
| **H4** 文体      | `genre` 与 d 的交互                                |
| **H5** JSD 倒 U | `info_theory` + 与 Creativity 的二次项（可接进 analyze） |
| **H6** 两阶段     | `mechanism` 高 d 子集对比                           |
| **H7** 双空间     | `d_H` 与 `d_L` 两套回归一致（需 Space-L 与 L 侧 d 写入流水线）  |


当前代码中 **H7 的 `d_L` 全量自动写入每条 main 记录** 可在你确认 Space-L 稳定后作为小改动接入；逻辑已在 `embedding_space.py` / `conflict.py` 中具备基础。

---

## 产出物清单（论文/复现）

- `data/generated/*.jsonl`：原始生成与中间字段  
- `data/generated/*_metrics.jsonl`：自动指标  
- `data/generated/*_judged.jsonl`：judge 分数（若跑）  
- `scripts/merge_judge_into_metrics.py`：把 `judge` 字段写回 `*_metrics.jsonl`（`analyze` 里 judge–自动指标相关表依赖此步）  
- `data/anchors.json`、`data/axis_vectors.npz`、`data/source_coords.jsonl`：HF 确定性坐标（见第 1a 步）  
- `results/flat.csv`、`results/tables/*.csv`、`results/figures/*.png`  
- `docs/proposal.md`、`docs/rubrics.md`：方案与量规全文

若你希望把本流程再缩成一页「给合作者的 checklist」，可以说明受众（仅工程 / 含导师），我可以再改一版语气与粒度。