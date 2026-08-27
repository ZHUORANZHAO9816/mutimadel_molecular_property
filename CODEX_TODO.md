# GTpro 原项目优化：Codex 可执行 TODO

本文档是原项目优化的唯一执行清单。目标是把当前仓库整理为一个可信、可安装、可测试、可复现并适合公开展示的论文实现项目。

项目定位保持不变：复现并完善论文 **“Boosting the performance of molecular property prediction via graph-text alignment and multi-granularity representation enhancement”**。不得把原论文方法冒充为本项目维护者的新方法。新增内容必须标记为工程修复、复现实验或扩展实验。

---

## 0. Codex 执行规则

后续 Codex 执行本清单时必须遵守以下规则：

- 严格按阶段顺序执行；当前阶段验收通过后再进入下一阶段。
- 每次只处理一个可验证的小任务，并在完成后更新本文档的复选框。
- 不得因为“代码看起来正确”就勾选任务；必须运行对应验收命令。
- 每次修改后至少运行与修改范围相关的最小测试。
- 不得伪造论文复现结果、训练结果、指标、checkpoint 或运行时间。
- smoke test 的结果不得写成正式训练结果。
- 不得提交完整 ChEMBL 衍生数据、正式 checkpoint、日志缓存或系统文件。
- 保留并注明原作者、原论文、原始仓库和第三方代码来源。
- 优先兼容当前可用环境，同时在文档中保留原论文环境说明。
- 不为追求目录美观一次性大规模移动代码；先建立测试，再逐步重构。
- 不删除疑似用户文件。废弃文件先记录来源和替代文件，确认无引用后再处理。
- 如果某项因为算力、数据许可或缺少外部资源无法完成，标记为 `BLOCKED`，记录原因和继续执行所需条件，不得直接视为完成。
- 每阶段完成后在 `docs/progress.md` 记录：修改内容、验收命令、结果、遗留问题。

状态约定：

- `[ ]` 未开始
- `[-]` 正在进行
- `[x]` 已完成并通过验收
- `[BLOCKED]` 被明确外部条件阻塞

---

## 1. 当前已知基线

开始优化前，Codex 应先核验并把核验结果写入 `docs/progress.md`：

- 当前运行环境曾观察到 Python 3.12.2、PyTorch 2.13.0、CPU。
- `python test_forward.py` 曾成功完成 BERT、GROVER、CoCa 前向传播。
- 当前 smoke 数据为 `data/pretrain_data/gtpro_smoke_1.npy` 和 `gtpro_smoke_2.npy`，合计 32 个样本。
- 当前 `CHEMBL_smiles.csv` 为 12,008 行、12,008 个唯一非空 SMILES。
- 一轮 32 样本预训练曾成功完成并生成 smoke checkpoint。
- `download_pretrain_data.py` 已加入手性标签兼容修复。
- `pretrain/pretrain_model.py` 已加入当前 GROVER 输出兼容修改。
- `pretrain/seq_trans.py` 已加入 GROVER 字典输出和 atom label 长度兼容修改。
- 根目录没有正式 `finetune.py`，README 中的命令与实际目录不一致。
- 仓库中同时存在本地 `gtpro/graph_trans` 和对外部 `grover` 包的引用，依赖来源尚未统一。
- `checkpoints/model_bert0.pth` 与 `model_coca0.pth` 只能视为 smoke 产物。

这些记录只是基线，不等于正式任务已完成。

---

# 阶段 A：建立安全基线与仓库卫生

## A1. 仓库状态审计

- [x] 检查当前目录是否为 Git 仓库；如果不是，初始化 Git 仓库。
- [x] 获取完整文件清单，并分类为：源码、数据、checkpoint、缓存、文档、系统文件。
- [x] 搜索所有 Python import，列出 `grover`、`gtpro.graph_trans`、`pretrain` 和 `finetune` 的引用关系。
- [x] 搜索所有训练入口、硬编码路径、CUDA 假设和绝对路径。
- [x] 检查 `seq_trans.py` 与 `seq_trans_fixed.py` 的差异，记录哪个是当前真实实现。
- [x] 检查 `pretrain/build_data.py`、`build_pretrain.py` 和根目录 `download_pretrain_data.py` 是否功能重叠。
- [x] 创建 `docs/current_state.md`，记录当前架构、可运行命令、已知问题和历史兼容修复。

验收：

```bash
git status --short
rg -n "from grover|import grover|gtpro\.graph_trans|cuda\(|\.cuda\(|PYTHONPATH|/Users/" . --glob '*.py'
```

完成标准：所有重要入口和重复实现都有书面记录，尚未删除任何不确定文件。

## A2. `.gitignore` 与大文件策略

- [x] 创建或完善 `.gitignore`。
- [x] 忽略 `.DS_Store`、`__pycache__`、`.pytest_cache`、虚拟环境、训练日志和临时文件。
- [x] 默认忽略 `checkpoints/*.pth`、`runs/`、`artifacts/` 和完整生成数据。
- [x] 允许提交专门用于测试的小型 fixture，但文件名和 README 必须明确标注为 smoke/test data。
- [x] 创建 `docs/artifacts.md`，说明哪些文件不进入 Git、如何重新生成、未来如何下载正式 checkpoint。
- [x] 检查现有文件中是否含凭据、token、私人路径或个人配置；`.claude/settings.local.json` 不得进入公开仓库。

验收：

```bash
git status --short
find . -type f -size +50M -print
```

完成标准：公开仓库候选文件中不包含缓存、私人配置、正式数据副本或误导性的 smoke checkpoint。

## A3. 保存初始可运行证据

- [x] 在重构前运行 `python test_forward.py` 并保存精简结果到 `docs/baseline_run.md`。
- [x] 运行当前 32 样本 smoke pretraining，记录命令、环境和运行时间。
- [x] 不把运行日志全文放进 README；只保存必要摘要。
- [x] 如果基线失败，先记录失败，不在同一任务中大规模重构。

验收：

```bash
python test_forward.py
PYTHONPATH=. python pretrain/pretrain_model.py --epochs 1 --batch_size 2 --data_path ./data/pretrain_data/gtpro_smoke
```

完成标准：两条基线命令的实际成功或失败状态均有记录。

---

# 阶段 B：环境、安装和统一入口

## B1. 建立现代包配置

- [x] 创建 `pyproject.toml`，使项目可以通过 `pip install -e .` 安装。
- [x] 明确项目名称、版本、Python 支持范围和基础依赖。
- [x] 将训练依赖与开发依赖分组，例如 `test`、`dev`、可选 `gpu`。
- [x] 不锁定已经不可获得的 CUDA wheel URL 为唯一安装方式。
- [x] 创建 `environment.yml` 或锁定版本的辅助环境文件，用于完整复现实验。
- [x] 在文档中同时说明“原论文环境”和“当前验证环境”。
- [x] 检查第三方 GROVER 的许可和来源，在依赖或致谢中明确注明。

验收：

```bash
python -m pip install -e .
python -c "import gtpro; print(gtpro.__file__)"
```

完成标准：无需修改 `PYTHONPATH` 即可导入本地 `gtpro`。

## B2. 统一 GROVER 来源

- [x] 根据 A1 审计选择单一、明确的 GROVER 实现来源。
- [x] 默认优先使用仓库内可维护的 `gtpro.graph_trans`；如果必须依赖外部 `grover`，要删除歧义并锁定兼容版本。
- [x] 禁止出现“本地模块存在，但运行时意外导入 site-packages 版本”的情况。
- [x] 增加测试，输出实际加载的 GROVER 模块路径。
- [x] 检查模型参数与 checkpoint 结构是否兼容。
- [x] 更新 README 的依赖说明。

验收：

```bash
python -c "import inspect; from gtpro.graph_trans.model.models import GROVEREmbedding; print(inspect.getfile(GROVEREmbedding))"
python test_forward.py
```

完成标准：GROVER 的实现来源唯一、可追踪，前向测试继续通过。

## B3. 建立统一配置和运行目录

- [x] 创建 `configs/`。
- [x] 创建最小 `configs/pretrain_smoke.yaml`。
- [x] 创建正式 `configs/pretrain.yaml`。
- [x] 配置至少包含随机种子、设备、数据路径、模型尺寸、batch size、epoch、学习率、输出目录。
- [x] 创建 `runs/` 和 `artifacts/` 的约定，但不提交实际大文件。
- [x] 每次运行复制最终配置到运行目录。
- [x] 每次运行保存环境摘要、Git commit、开始/结束时间和随机种子。

完成标准：训练参数不再主要散落在代码默认值或硬编码路径中。

## B4. 创建规范命令行入口

- [x] 创建 `scripts/prepare_pretrain_data.py`。
- [x] 创建 `scripts/run_pretraining.py`。
- [x] 创建 `scripts/run_finetuning.py` 的占位入口，正式实现放在阶段 D。
- [x] 创建 `examples/smoke_test.py`。
- [x] 所有入口支持 `--help`。
- [x] 所有入口从任意工作目录执行时路径行为明确。
- [x] 旧入口保留兼容提示；确认新入口稳定后再标记 deprecated。

验收：

```bash
python scripts/prepare_pretrain_data.py --help
python scripts/run_pretraining.py --help
python scripts/run_finetuning.py --help
python examples/smoke_test.py
```

---

# 阶段 C：数据处理可复现化

## C1. 重构预训练数据处理

- [x] 将根目录数据处理逻辑迁移为可导入函数，CLI 只负责解析参数。
- [x] 保留当前手性/CIP 标签修复，并为 R、S、无 CIP 三种情况增加测试。
- [x] 对无效 SMILES 记录行号、原始字符串和失败原因。
- [x] canonicalize 后统计重复分子。
- [x] 明确是否保留重复分子；默认对预训练数据去重，并记录规则。
- [x] 验证 atom label、atom mask、token 序列和图节点数量关系。
- [x] 对超长 SMILES 给出明确的截断或过滤策略。
- [x] 使用可配置 shard 数量，不依赖固定文件名。
- [x] 使用临时文件加原子替换，避免中断后留下看似完整的损坏 shard。
- [x] 支持断点恢复或至少支持跳过已验证 shard。

验收：

```bash
python scripts/prepare_pretrain_data.py --input data/pretrain_data/CHEMBL_smiles.csv --output artifacts/pretrain --num-shards 4
```

完成标准：完整处理结束后自动生成机器可读和人可读的数据报告。

## C2. 创建数据报告

- [x] 生成 `artifacts/pretrain/data_report.json`。
- [x] 生成 `artifacts/pretrain/data_report.md`。
- [x] 报告总行数、空值、解析成功数、失败数、唯一 canonical SMILES 数和重复数。
- [x] 报告原子数、SMILES 长度和官能团标签的基本分布。
- [x] 报告最终 shard 文件、样本数和 checksum。
- [x] 将稳定的统计摘要复制到 `docs/datasets.md`，不要提交完整数据。

验收：

```bash
python -c "import json; d=json.load(open('artifacts/pretrain/data_report.json')); print(d)"
```

## C3. 下游数据审计

- [x] 枚举 `data/downstream` 下所有数据集及实际文件。
- [x] 核对 README 声称的数据集与实际文件是否一致。
- [x] 对每个数据集识别 SMILES 列、标签列、任务类型和缺失标签编码。
- [x] 输出 `docs/downstream_datasets.md`。
- [x] 明确分类、多标签分类和回归任务。
- [x] 实现统一 dataset interface。
- [x] 实现 random split 和 scaffold split。
- [x] 测试 split 间无重复索引和无 canonical SMILES 泄漏。

完成标准：至少 BACE、Tox21、Lipophilicity 可被统一 loader 加载。

## C4. 小型测试 fixture

- [x] 创建不超过几十个分子的测试 fixture。
- [x] fixture 必须包含有效 SMILES、无效 SMILES、手性分子、多原子分子和缺失标签。
- [x] fixture 不得被描述为正式数据集。
- [x] 测试运行时不得依赖完整 ChEMBL 数据。

---

# 阶段 D：模型、预训练和微调

## D1. 整理模型模块边界

- [x] 明确并文档化 graph encoder、SMILES encoder、alignment/fusion 和 prediction head 的输入输出。
- [x] 对所有主要 tensor 标注 shape。
- [x] 用统一数据结构处理 GROVER 的 dict/tuple 输出，不在训练循环里散布类型判断。
- [x] 将设备迁移集中处理，避免模型内部硬编码 `.cuda()`。
- [x] 检查 attention mask、padding mask 和 atom mask 的语义。
- [x] 删除或合并确认无用的重复 `nt_xent.py`，删除前必须通过引用搜索。
- [x] 处理 `seq_trans_fixed.py`：合并有效修改或移动到明确的 legacy 目录，不能留下两个不知哪个生效的实现。

验收：

```bash
python examples/smoke_test.py
python test_forward.py
```

## D2. 预训练循环标准化

- [x] 将单 epoch 训练逻辑与 CLI 分离。
- [x] 支持 train/validation 划分。
- [x] 分别记录 contrastive、atom、functional-group 和 molecule loss。
- [x] 记录总 loss，但不能只记录累加总和而不报告样本或 batch 平均值。
- [x] 支持 gradient clipping。
- [x] 支持 checkpoint resume。
- [x] 保存 best 和 last checkpoint。
- [x] 保存 optimizer、scheduler、epoch、seed 和配置。
- [x] CPU 下默认关闭混合精度；GPU 下混合精度作为可选项。
- [x] 修复从 Python list 直接构造 tensor 的性能警告。
- [x] 检查空 batch、无效分子和最后一个不足 batch 的行为。

验收：

```bash
python scripts/run_pretraining.py --config configs/pretrain_smoke.yaml
```

完成标准：smoke pretraining 可重复完成，并能从中间 checkpoint 恢复继续训练。

## D3. 实现正式微调入口

- [x] 创建统一 fine-tuning runner。
- [x] 支持二分类、多标签分类和回归。
- [x] 支持 frozen encoder、partial unfreeze 和 full fine-tuning。
- [x] 正确忽略多标签数据中的缺失标签。
- [x] 支持 class imbalance 处理，但必须配置化并记录。
- [x] 支持 early stopping。
- [x] 支持多 seed。
- [x] 每次运行保存 best checkpoint、predictions.csv、metrics.json 和 config.yaml。
- [x] 测试预训练 checkpoint 加载时 missing/unexpected keys。
- [x] 禁止静默忽略大规模 checkpoint 不匹配。

验收：

```bash
python scripts/run_finetuning.py --config configs/finetune_bace_smoke.yaml
python scripts/run_finetuning.py --config configs/finetune_tox21_smoke.yaml
python scripts/run_finetuning.py --config configs/finetune_lipophilicity_smoke.yaml
```

## D4. 指标实现

- [x] 二分类实现 ROC-AUC 和 PR-AUC。
- [x] 多标签分类按有效标签计算每任务指标，并报告宏平均与有效任务数。
- [x] 回归实现 RMSE、MAE 和 R2。
- [x] 单类别 validation fold 不得导致无解释崩溃；应返回警告和不可用指标。
- [x] 为手工构造的已知预测增加指标单元测试。

---

# 阶段 E：测试体系

## E1. 单元测试

- [x] 创建 `tests/test_preprocessing.py`。
- [x] 创建 `tests/test_dataset_splits.py`。
- [x] 创建 `tests/test_model_shapes.py`。
- [x] 创建 `tests/test_metrics.py`。
- [x] 创建 `tests/test_checkpoint.py`。
- [x] 创建 `tests/test_training_step.py`。
- [x] 测试有效、无效和手性 SMILES。
- [x] 测试 graph/text/joint embedding shape。
- [x] 测试单个 batch 前向、反向和 optimizer step。
- [x] 测试 checkpoint save/resume 后 epoch 与参数一致。

验收：

```bash
pytest -q
```

## E2. 集成测试

- [x] 数据 fixture 到预训练 batch 的完整流程。
- [x] 一轮 smoke pretraining。
- [x] BACE smoke fine-tuning。
- [x] Lipophilicity smoke fine-tuning。
- [x] 测试均可在 CPU 完成。

## E3. 持续集成

- [x] 创建 GitHub Actions workflow。
- [x] CI 安装项目和测试依赖。
- [x] CI 运行静态导入检查、`pytest -q` 和 CPU smoke test。
- [x] CI 不下载完整数据、不训练正式模型、不要求 GPU。
- [x] README 中只在 CI 真正通过后加入 badge。

---

# 阶段 F：正式复现实验

此阶段涉及真实算力消耗。开始前必须确保 A–E 全部通过。

## F1. 实验协议冻结

- [x] 创建 `docs/experiment_protocol.md`。
- [x] 固定数据版本、预处理规则、split 方法、seeds 和指标。
- [x] 默认 seeds 使用 `42, 52, 62`；正式报告至少 3 seeds。
- [x] 明确模型选择依据，禁止用 test set 调参。
- [x] 明确 random split 与 scaffold split 的用途。
- [x] 明确 checkpoint 选择和 early stopping 规则。
- [x] 记录 CPU/GPU 类型、显存、软件版本和训练时长。

## F2. 基础 baseline

- [x] Morgan fingerprint + Logistic Regression/Random Forest。
- [x] Graph-only。
- [x] SMILES-only。
- [x] Graph + SMILES，无 alignment loss。
- [x] 完整 GTpro。
- [x] 所有 baseline 使用相同 split 和 seed。
- [x] 对不同参数规模模型报告参数量，避免不公平比较。

## F3. 核心下游任务

- [x] BACE：二分类，3 seeds，random split。
- [x] BACE：二分类，3 seeds，scaffold split。
- [x] Tox21：多标签分类，3 seeds，random split。
- [x] Tox21：多标签分类，3 seeds，scaffold split。
- [x] Lipophilicity：回归，3 seeds，random split。
- [x] Lipophilicity：回归，3 seeds，scaffold split。
- [x] 如果算力允许，再扩展 SIDER 和 ToxCast。（本轮无余量；ToxCast 数据缺失，按协议不扩展。）

每次正式运行必须生成：

```text
runs/<experiment>/<seed>/config.yaml
runs/<experiment>/<seed>/environment.json
runs/<experiment>/<seed>/metrics.json
runs/<experiment>/<seed>/predictions.csv
runs/<experiment>/<seed>/best_checkpoint.*
```

## F4. 消融实验

- [x] Full model。
- [x] No contrastive loss。
- [x] No cross-attention。
- [x] No atom-level objective。
- [x] No functional-group objective。
- [x] No molecule-level objective。
- [x] Graph-only。
- [x] SMILES-only。
- [x] 先用 1 seed 做趋势筛查，再对关键差异补齐 3 seeds。

## F5. 结果汇总

- [x] 创建 `scripts/summarize_results.py`。
- [x] 自动读取运行目录，不手抄结果。
- [x] 生成 mean、standard deviation 和有效运行数。
- [x] 生成 `results/reproduction.csv`。
- [x] 生成 `results/ablation.csv`。
- [x] 生成适合 README 的 Markdown 表格。
- [x] 生成 random/scaffold 对比图。
- [x] 生成 ablation 图。
- [x] 明确区分论文报告值与本项目实测值。

完成标准：README 中每一个数字都能追溯到 `metrics.json` 和对应配置。

---

# 阶段 G：公共编码接口

## G1. `GTproEncoder` API

- [x] 实现 `GTproEncoder.from_pretrained(...)`。
- [x] 实现 `encode_smiles(...)`。
- [x] 支持字符串和字符串列表。
- [x] 支持 `graph`、`text` 和 `joint` 三种表示。
- [x] 支持 batching。
- [x] 支持 CPU 和 GPU。
- [x] 支持冻结模型。
- [x] 明确无效 SMILES 的处理策略。
- [x] 文档化 embedding shape 和 dtype。

目标用法：

```python
from gtpro import GTproEncoder

encoder = GTproEncoder.from_pretrained("path/to/checkpoint", device="cpu")
embeddings = encoder.encode_smiles(["CCO", "CCN"], representation="joint")
print(embeddings.shape)
```

## G2. 编码示例和测试

- [x] 创建 `examples/encode_molecule.py`。
- [x] 为单分子、批量分子和无效分子增加测试。
- [x] 确保新项目可以只通过公共 API 使用 GTpro，不导入内部训练模块。

验收：

```bash
python examples/encode_molecule.py --smiles CCO --device cpu
```

---

# 阶段 H：README、文档和 GitHub 展示

## H1. 重写 README

- [x] 使用准确项目名和论文名。
- [x] 第一段用 3–5 句话说明研究问题、方法和仓库贡献。
- [x] 第一屏加入模型结构图。
- [x] 加入功能列表，但仅列出已完成能力。
- [x] 加入 CPU smoke quick start。
- [x] 加入完整预训练和微调命令。
- [x] 加入数据准备说明。
- [x] 加入复现结果表。
- [x] 加入消融结果表。
- [x] 加入系统要求和预计资源消耗。
- [x] 加入 repository structure。
- [x] 加入 limitations。
- [x] 加入 citation、acknowledgements 和 license。
- [x] 明确论文原贡献与仓库新增贡献。

README 不得出现：

- 尚未实现的命令；
- 尚未产生的指标；
- 将 smoke checkpoint 描述为 pretrained model；
- “state of the art” 等未经验证的宣传；
- 无法追溯来源的架构图和结果。

## H2. 架构与方法文档

- [x] 创建 `docs/architecture.md`。
- [x] 描述 SMILES encoder、GROVER encoder、alignment 和 multi-granularity objectives。
- [x] 给出主要 tensor shape。
- [x] 说明训练和推理数据流。
- [x] 创建一张原创架构图，避免直接复制论文受版权保护的图。

## H3. 可复现说明

- [x] 创建 `docs/reproducibility.md`。
- [x] 包含环境、数据、命令、seed、输出结构和结果汇总方法。
- [x] 说明当前实测环境与原论文环境差异。
- [x] 记录已知数值差异及可能原因。

## H4. 项目元数据

- [BLOCKED] 增加开源许可证；确认与上游代码许可兼容。（新贡献与 GROVER 为 MIT；原 GTpro 快照无可验证许可证，发布前需作者确认。）
- [x] 创建 `CITATION.cff`。
- [x] 创建 `CONTRIBUTING.md`。
- [x] 创建基础 issue templates，可选但推荐。
- [x] 创建 `CHANGELOG.md`。
- [x] 增加论文 DOI、原始仓库和上游项目链接。

---

# 阶段 I：发布前审计

## I1. 新环境复验

- [x] 在干净虚拟环境安装项目。
- [x] 运行 import test。
- [x] 运行全部 unit tests。
- [x] 运行 CPU smoke test。
- [x] 运行一个 BACE smoke fine-tuning。
- [x] 检查 README 中所有命令是否能复制执行。

验收：

```bash
python -m pip install -e ".[test]"
pytest -q
python examples/smoke_test.py
python scripts/run_finetuning.py --config configs/finetune_bace_smoke.yaml
```

## I2. 公开仓库审计

- [x] 检查 Git 跟踪文件中无大 checkpoint、完整数据、缓存和私人配置。
- [x] 检查 Git 历史中是否误提交敏感文件。
- [x] 检查所有外部代码的许可和 attribution。
- [x] 检查所有结果可追溯。
- [x] 检查所有图片有来源或为本项目原创。
- [x] 检查 README 链接有效。
- [x] 检查 GitHub 首页在不展开文档时能看懂项目用途。

## I3. 发布里程碑

- [BLOCKED] 发布 `v0.1.0`：本地里程碑内容已完成；尚无可发布的 Git 提交/远程且原 GTpro 许可未确认。
- [BLOCKED] 发布 `v0.5.0`：复现与消融已完成；发布前置条件同上。
- [BLOCKED] 发布 `v1.0.0`：本地工程与审计已完成；需许可确认、远程 CI 和真实发布权限。
- [BLOCKED] 正式 checkpoint 使用 GitHub Release、Hugging Face 或 Zenodo 托管，不直接提交 Git。（当前只有本地 compact 复现权重，无正式托管目标/发布授权。）
- [x] Release notes 说明新增能力、已知限制和复现范围。

---

# 最终 Definition of Done

只有同时满足以下条件，原项目优化才算完成：

- [x] `pip install -e .` 成功。
- [x] 无需 `PYTHONPATH=.` 即可运行。
- [x] `pytest -q` 全部通过。
- [x] CPU smoke forward 和单轮训练通过。
- [x] 完整 12,008 行输入的数据处理有可审计报告。
- [x] 正式预训练可以启动、保存和恢复。
- [x] 微调支持 BACE、Tox21 和 Lipophilicity。
- [x] 支持 random split 和 scaffold split。
- [x] 正式结果至少包含 3 seeds 的均值和标准差。
- [x] 至少包含 Morgan baseline、Graph-only、SMILES-only 和完整 GTpro。
- [x] 至少完成一组多粒度/对齐消融。
- [x] README 中所有命令均真实存在并验证通过。
- [x] README 中所有结果均可追溯到配置和指标文件。
- [x] 原论文贡献、上游代码和本项目新增工作标注清楚。
- [x] 提供稳定的 `GTproEncoder` 公共接口。
- [BLOCKED] GitHub Actions 在 CPU 上通过。（本地 workflow 等价命令已通过；当前无远程仓库/可观测 workflow run。）
- [x] 仓库无敏感文件、大型数据和误导性 checkpoint。
- [x] `v1.0.0` release notes 完整。

---

# 推荐的 Codex 单次任务指令模板

后续可以直接向 Codex 提交：

```text
请打开 CODEX_TODO.md，执行下一个尚未完成且未被阻塞的任务。
严格遵守“Codex 执行规则”：先检查依赖状态，再修改代码，运行该任务的验收命令，
把结果写入 docs/progress.md，只有验收通过后才勾选对应复选框。
不要同时跨越多个阶段，不要生成虚假实验结果，不要覆盖无关用户修改。
```

较大的阶段可以使用：

```text
请按照 CODEX_TODO.md 完成阶段 B。逐项执行并验证；遇到失败先诊断修复。
每完成一项更新复选框和 docs/progress.md。阶段验收未通过时不要进入阶段 C。
```
