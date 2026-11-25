# Evolutionary Prompt Search 服务器迁移与 Codex 交接

交接日期为 2026-11-25。项目用固定的 Qwen3.5-9B checkpoint 做文本分类和候选生成，两种用途各自使用推理配置。当前工作重点是 **DepSign 三分类的全组件错例反馈进化**。这项扩展待实现。代码和汇总结果的交接基线为 `main` 上的 `a6a774a`（`server transfer`）。

## 新对话从这里开始

在新服务器上打开项目根目录，把这句话发给 Codex，并附上本文或让它直接读取项目里的文件。

> 读 CODEX_HANDOFF.md，检查迁移后的文件和环境。从“当前要接着做的任务”继续，逐步实现 DepSign 三分类的全组件错例反馈进化。先做本地准备检查和小规模验证，再开始正式搜索。使用已有实验作为对照，按实际结果报告每代 best 和 mean macro-F1。

## 当前要接着做的任务

用户最后要求“做包括所有 component 的进化”，指刚完成的 G12 错例反馈实验。把错例反馈从只改 `instruction` 扩展到同一个个体的 **`task_definition`、`label_definition`、`instruction`**。项目已有按父代混淆矩阵编辑这三个组件的 GA，可复用它的组合、校验和报告。逐条错例反馈与三组件编辑的接入尚未开始。

1. **复用现有三分类初始文件。** `prompts/depsign/components/three_class/` 下三个 JSON 提供任务定义、标签定义与示例、初始指令。每个候选是一套完整组合，在同一个 `dev_100.tsv` 上评分。标签键、评估数据的真实标签、输出约定和 few-shot 的文本、标签、ID、顺序固定。
2. **把父代错例接入组件编辑。** 现有反馈实验从父代在搜索集的预测中抽最多 3 条错例，每个真实类别最多 1 条，向本地 Qwen 提供截短文本、真实标签和预测标签。借用这个上限与采样逻辑。三个组件都要能得到有针对性的编辑。标签定义沿用 creative operator 的研究设定，允许根据错例调整或重写判定规则，保留输出类别键。编辑结果的成因和语义质量仍需要验证。
3. **用完整组合做精英保留。** 以搜索集 macro-F1 对父代和有效子代统一排序，保留前 N，同分按创建顺序。按完整组合精确去重。参考 G12 方法，固定代数并报告全部种子，画每代实际保留种群的 best 和 mean macro-F1，记录有效新子代数、失败和实际模型调用次数。理想走势是两条曲线总体上升、允许短平台、后期趋稳。已有 G12 分析用 G9–G12 两条线的变化范围各 ≤0.005，且末三代每代有新子代完成评估，作为平台判据。
4. **先小规模验证，再固定正式比较设置。** 检查三个字段均可变化、few-shot 完全相同、错例只来自搜索集，验证组合评分、去重与留存。需要把组件编辑方式、父代选择、请求数、种群大小和种子写入新运行配置。若想量化扩展到三组件本身的效果，尽量与 instruction-only 对照匹配初始组合、搜索样本、种子和提案预算，同时报告生成请求数差异。此前用过 11、29、47 三个种子。
5. **冻结后做额外评估。** `dev_100` 同时提供错例和搜索分数，因此其 F1 衡量优化过程。冻结胜出组合后再用另外 300 条开发帖评估，确认数据的标签和错例只用于最终评分。这 300 条与当前搜索集、few-shot 和已知 smoke 样本分开，但来自此前项目曾用过的清洗开发集。正式 test 成绩仍待单独验证。

已有文本 DE 只进化了 `instruction`。三组件 DE 也是未完成的独立实验。如果用户明确要将 DE 用于这三个组件，可在三组件评估入口上单独加入，并与错例反馈方案同预算比较。

## 可复用的代码

| 路径 | 用途 |
|---|---|
| [README.md](README.md) | 常规入口、配置、已实现的 DepSign、Dreaddit 和 joint 用法 |
| [experiment/README.md](experiment/README.md) | 复跑对照、错例反馈和 G12 的命令 |
| [experiment/feedback.py](experiment/feedback.py) | 取父代错例、分两次请求批评和改写，目前返回一个 instruction |
| [experiment/evaluation.py](experiment/evaluation.py) | 冻结搜索集与 300 条确认集，计算 macro-F1 并返回逐条预测 |
| [scripts/component_ga.py](scripts/component_ga.py) | 三组件 GA 的生成、完整组合评估和留存 |
| [scripts/component_prompts.py](scripts/component_prompts.py) | 校验固定标签键和长度、拼装分类模板、检查示例重叠 |
| [experiment/component_search.py](experiment/component_search.py) | 用 `--survivors elite` 运行三组件 GA 的独立入口，会写有效配置快照 |
| [experiment/curve_report.py](experiment/curve_report.py) | 从实际保留名单和候选分数重算并绘制 G12 曲线 |
| [configs/experiment_v1.json](configs/experiment_v1.json) | 包含 `search`、`component_search`、模型设置等实验参数 |

## 已经跑过的实验

以下是 DepSign 三分类开发集实验。不同实验的种群大小和初始指令有差异。

| 实验 | 实测结果 | 记录 |
|---|---|---|
| Instruction-only 错例反馈＋精英保留，N=5，3 个种子各 12 代 | 初始 best／mean 均为 0.3745／0.3596。G12 在种子 11 为 0.4255／0.4166，29 为 0.4434／0.4279，47 为 0.4314／0.4178。11 和 47 达到预设平台标准，29 在 G12 创新高。 | [三种子曲线](experiment/results/feedback_curve_g12.png) · [逐代值](experiment/results/feedback_curve_g12.json) |
| Instruction-only 的 GA、文本 DE 与独立提案对照，N=5，3 个种子各 4 代 | DE 三个种子都在 G1 达到各自四代最高分。60 个槽位中 28 个重复。DE 获胜 prompt 平均搜索 F1 为 0.4038，在额外 300 条开发帖上为 0.3626，初始最佳为 0.3700。 | [四组对照](experiment/results/instruction_results.md) |
| 已有三组件 GA，矩阵反馈，N=10，种子 11，4 代 | 精英保留 G4 best／mean 为 0.4278／0.3971，锦标赛为 0.4211／0.3835。精英组获胜组合三个字段均曾变化。300 条确认帖上两组分别为 0.3907 和 0.3969，配对差值区间跨零。 | [组件对照](experiment/results/component_findings.md) · [曲线](experiment/results/component_fitness.png) |

G4 和 G12 的错例反馈搜索都已完成并冻结胜出者。这两组的 300 条确认评估仍待完成，原服务器的 `127.0.0.1:8000` 当时拒绝连接。四组方法的确认评估和三组件矩阵反馈的确认评估已有结果。相关协议和复跑命令见 [experiment/README.md](experiment/README.md)。

## 迁移到新服务器

**代码和汇总图**随 Git 仓库获取。此前 `a6a774a` 已在 `origin/main`。新生成的交接文件需要随项目另行复制，或者由你提交后在新服务器拉取。Git 忽略或未实质跟踪的文件需要另外搬。

| 路径 | 什么时候需要 |
|---|---|
| `LLM/Qwen3.5-9B/` | 本地 tokenizer 准备检查和推理，约 19 GiB。复制完整目录，包括模型分片、索引和 tokenizer 文件 |
| `data/processed/depsign/dev_100.tsv`、`dev_100.manifest.json`、`dev_v1/dev_clean.tsv` | 恢复相同搜索和确认划分，三者都需要。DepSign 搜索集为 41／51／8，抽样种子 1729 |
| `experiment/runs/` | 保留冻结配置、历史预测和实验续跑，约 14 MiB。尤其保留 `instruction_suite/`、`feedback_suite/`、`feedback_g12/`、`components_elite_g4/` |
| `data/processed/depsign/component_ga_runs/`、`data/processed/depsign/ga_runs/` | 追溯历史实验。组件对照脚本默认读取其中的 `component_ga_runs/three_class_20261125T012800064686Z/performance.json` |
| `data/processed/dreaddit/dre_dev_100.tsv` 和对应 `.manifest.json` | 继续 Dreaddit 或 DepSign＋Dreaddit 双目标搜索。Dreaddit 样本来自 train，0／1 为 47／53 |
| `data/external/` | 重新预处理时恢复自有的原始数据。DepSign 的原始目录在 Git 中是没有 `.gitmodules` 映射的 gitlink，clone 不会填入 TSV |

已有服务器通过 SSH 直接传文件时，可以从旧项目根目录执行下例。先在新服务器检出仓库，并将 `new-host:~/...` 替换成目标地址和项目路径。这条命令覆盖最低限度的 DepSign 续接文件和实验记录；其他文件按上表选择。数据和原始运行记录只复制到你的私有服务器。

```bash
rsync -avR ./CODEX_HANDOFF.md ./LLM/Qwen3.5-9B/ \
  ./data/processed/depsign/dev_100.tsv \
  ./data/processed/depsign/dev_100.manifest.json \
  ./data/processed/depsign/dev_v1/dev_clean.tsv \
  ./experiment/runs/ \
  new-host:~/Evolutionary-Prompt-Search-for-Mental-Health-Classification/
```

### 环境与首个检查

从新服务器上的项目根目录执行，准备检查使用本地 tokenizer，并把检查输出放在指定的新目录。

```bash
./setup_env.sh
conda activate prompt-search
export PROMPT_SEARCH_CONTEXT_TOKENS=8192
python -B scripts/run_ga.py --task depsign --search-space components \
  --label-mode three_class --generations 1 --prepare-only \
  --output-dir experiment/runs/components_migration_check
```

`setup_env.sh` 创建 Python 3.12.15 环境。原项目固定 vLLM 0.32.0、Transformers 5.19.0、PyTorch 2.13.0+cu132 和作图库，见 [requirements-prompt-search.txt](requirements-prompt-search.txt)。在没有 GPU 访问的节点，可用 `./setup_env.sh --skip-cuda-check` 安装，再到分配了 GPU 的节点做推理检查。

在有可用 GPU 的终端启动服务。该脚本固定选择 GPU 0，选择其他 GPU 或复现旧服务签名时，使用下一段的显式命令。客户端也使用 8192。服务起来后可执行 `curl -fsS http://127.0.0.1:8000/v1/models` 检查。

```bash
PROMPT_SEARCH_CONTEXT_TOKENS=8192 ./start_vllm.sh
```

### 继续已冻结的旧运行

旧运行要求数据、关键源码、模型元数据、推理设置和 vLLM 服务签名一致，还检查权重分片文件的大小和纳秒级 mtime。保留原始文件字节和时间戳，迁移后核对 `experiment/runs/instruction_suite/protocol.json` 中的 `weight_files`。如果两端的 rsync 均支持纳秒时间比较，可在传输命令加上 `--modify-window=-1`，并用下面的离线准备检查验证。脚本默认服务会把模型目录作为绝对路径传给 vLLM，而旧实验记录的服务模型根路径是 `./LLM/Qwen3.5-9B`。如果需要继续确认评估，先在项目根目录启动匹配的服务。下例的 `0` 换成你有权使用的 GPU 索引。

```bash
conda activate prompt-search
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
  vllm serve ./LLM/Qwen3.5-9B --served-model-name qwen35-9b \
  --host 127.0.0.1 --port 8000 --dtype bfloat16 --tensor-parallel-size 1 \
  --max-model-len 8192 --max-num-seqs 1 --gpu-memory-utilization 0.90 \
  --language-model-only --enforce-eager --generation-config vllm \
  --reasoning-parser qwen3 --default-chat-template-kwargs '{"enable_thinking": false}'
```

先在客户端执行下面两项离线检查。第一条核对冻结协议及 G4 记录。第二条用保存的逐代候选重绘 G12 曲线。确需完成旧确认时，等服务运行后依次执行 `python -B experiment/feedback.py --stage confirm` 和 `python -B experiment/extend_feedback.py --stage confirm`。遇到协议不匹配，先调查文件或服务签名，用新的输出目录跑新设定，保留旧收据和哈希。

```bash
export PROMPT_SEARCH_CONTEXT_TOKENS=8192
python -B experiment/extend_feedback.py --stage prepare
python -B experiment/curve_report.py --suite experiment/runs/feedback_g12 \
  --output experiment/runs/migration_plot_check
```

## 保留的研究约定

- DepSign 二分类把 `moderate` 和 `severe` 合为 depression。三分类保留原有三类。分类设置固定为 temperature 0、seed 0、thinking 关闭。模型权重保持冻结。
- EvoPrompt 是进化搜索方法来源。ProTeGi 提供基于错例生成文字反馈的先例。Arnaout 等人的工作启发 validity- and reliability-inspired initialization。分类 F1 测量分类表现，框架中与真实结果关联、时间可靠性和跨人群可靠性需要独立实验。相关来源见 [references.bib](references.bib) 和 [实验协议](experiment/protocol.md)。
- 保留各方法的实际设置、种子、模型调用数和结果，避免拿不同预算的 best 直接归因给算法。C-SSRS 跨任务验证和 DepSign 同任务跨数据集验证是独立研究问题，后者所需的 Dataset B 尚未确定。
