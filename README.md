# SMILES-GUI

A Streamlit-based graphical interface for molecular SMILES input, validation, and single-step retrosynthesis (inverse synthesis) prediction using deep learning (PyTorch) and RDKit.

Graduation Project – Shanghai University of Health Sciences  
Author: Wenhao Ju (Ju Wenhao)

[Live Demo (Public Deployment)](https://smiles-reactor-juwenhao.streamlit.app)

---

## Table of Contents

- [Live Demo](#live-demo)
- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage (GUI Inference)](#usage-gui-inference)
- [Model Training](#model-training)
- [Data & Checkpoints](#data--checkpoints)
- [Model Caching](#model-caching)
- [Checkpoint Conversion & Safety](#checkpoint-conversion--safety)
- [Batch Conversion Tool](#batch-conversion-tool)
- [Exporting Predictions](#exporting-predictions)
- [Configuration](#configuration)
- [Logging & Reproducibility](#logging--reproducibility)
- [Performance & Evaluation](#performance--evaluation)
- [Deployment](#deployment)
- [Roadmap](#roadmap)
- [Limitations](#limitations)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)
- [Contact](#contact)

---

## Live Demo

You can try the application instantly (no installation required):

👉 <https://smiles-reactor-juwenhao.streamlit.app>

The hosted version:

- Loads a pre-trained checkpoint from the server environment.
- Allows entering a target product SMILES and viewing predicted reactant sets.
- Renders molecules with RDKit.
- May have limited computational resources vs. local GPU inference.

If the app sleeps (resource limits), reload to restart.

---

## Overview

SMILES-GUI provides an interactive environment to:

1. Enter a target molecule as a SMILES string.  
2. Run a trained retrosynthesis model to propose precursor/reactant candidates.  
3. Visualize predicted disconnections and structures.  
4. (Optionally) retrain the model locally with custom data.

Core technologies:

- RDKit for SMILES parsing and depiction
- PyTorch for sequence / structural model inference
- Streamlit for lightweight browser-based UI

---

## Features

| Category | Description |
|----------|-------------|
| SMILES Input | Free-form input box with basic validation |
| Retrosynthesis | Single-step inverse synthesis prediction |
| Visualization | RDKit-rendered product and predicted reactants |
| Model Checkpoints | Loaded from `experiments/` directory |
| Training Script | `train.py` for supervised training |
| Dataset Integration | Uses datasets stored under `data/` |
| Extensibility | Modular structure for adding new models or decoders |
| Deployment Ready | Works locally and via Streamlit Cloud |
| English-Only UI | Simplified for international usage |

---

## Architecture

High-level workflow:

```text
User Input (SMILES) → Validation (RDKit) → Tokenization / Encoding → Model Inference (PyTorch) →
Candidate Decoding (beam / top-k) → Visualization & Ranking → GUI Output
```

Conceptual modules:

- Data pipeline (loading, normalization)
- Model (graph/sequence encoder + edit decoder)
- Inference utilities (beam search)
- Visualization (RDKit drawing)
- Frontend (`home.py` Streamlit app)

---

## Project Structure

Simplified layout (actual may differ):

```text
SMILES-GUI/
├─ home.py
├─ train.py
├─ requirements.txt
├─ pages/
│  ├─ 1selection.py
│  ├─ 2prediction.py
├─ models/
│  ├─ graph2edits.py
│  ├─ beam_search.py
├─ utils/
│  ├─ rxn_graphs.py
│  ├─ chem.py
├─ experiments/
│  ├─ uspto_50k/
│  │  ├─ with_rxn_class/
│  │  └─ without_rxn_class/
├─ data/
│  └─ uspto_50k/
└─ convert_checkpoint.py
```

---

## Installation

### Prerequisites

- Python 3.9.6
- (Recommended) Conda (for RDKit)
- (Optional) CUDA-capable GPU

### Create Environment

```bash
conda create -n smiles-gui python=3.9.6 -y
conda activate smiles-gui
conda install -c conda-forge rdkit -y
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Quick Start

```bash
streamlit run home.py
```

Enter a SMILES (e.g. `CCOC(=O)C1=CC=CC=C1`) → Select a model → Predict.

---

## Usage (GUI Inference)

```bash
streamlit run home.py
```

UI Elements:

- Product SMILES input
- Model selection with version tags
- Predict button
- Results table (Reactant 1 / Reactant 2 / Confidence)
- Download buttons (CSV / SDF)

---

## Model Training

```bash
python train.py \
  --data_dir data/processed \
  --train_file train.csv \
  --val_file val.csv \
  --output_dir experiments/checkpoints \
  --epochs 50 \
  --batch_size 128 \
  --lr 1e-4 \
  --device cuda
```

| Argument | Description |
|----------|-------------|
| `--data_dir` | Root processed dataset directory |
| `--train_file` | Training split filename |
| `--val_file` | Validation split filename |
| `--test_file` | (Optional) test split filename |
| `--output_dir` | Where to save checkpoints |
| `--epochs` | Training epochs |
| `--batch_size` | Batch size |
| `--lr` | Learning rate |
| `--device` | `cuda` or `cpu` |
| `--seed` | Random seed |
| `--beam_size` | (Optional) beam width for validation decoding |

---

## Data & Checkpoints

| Path | Purpose |
|------|---------|
| `data/raw/` | Original unmodified dataset(s) |
| `data/processed/` | Tokenized / standardized training artifacts |
| `experiments/checkpoints/` | Saved model weights |
| `experiments/logs/` | Training event / metric logs |
| `experiments/configs/` | Configuration files (if used) |

Recommendations:

- Keep large raw datasets out of version control.
- Save each run's config & seed.

---

## Model Caching

The GUI caches the last loaded model in `st.session_state._model_cache` to avoid repeated deserialization.  
If the selected checkpoint path is unchanged, only inference runs—saving time especially on CPU.

---

## Checkpoint Conversion & Safety

PyTorch 2.6+: `torch.load` defaults to `weights_only=True` (safer).  
Older checkpoints with pickled custom classes fallback to full pickle.

Safe format (`format_version = 2`):

```python
{
  'format_version': 2,
  'model_args': {
      'config': {...},
      'atom_vocab_list': [...],
      'bond_vocab_list': [...]
  },
  'state': <state_dict>
}
```

Convert single file:

```bash
python convert_checkpoint.py \
  --input experiments/uspto_50k/without_rxn_class/.../epoch_120.pt \
  --output experiments/uspto_50k/without_rxn_class/.../epoch_120_safe.pt \
  --verify
```

---

## Batch Conversion Tool

单文件与批量转换已统一为同一个脚本 `convert_checkpoint.py`。旧的 `batch_convert_checkpoints.py` 包装器已移除（请直接使用统一脚本）。

递归转换全部 legacy / v1 检查点：

```bash
# 仅预览（不会真正写入）
python convert_checkpoint.py --batch-root experiments/uspto_50k --dry-run

# 实际转换（单进程）
python convert_checkpoint.py --batch-root experiments/uspto_50k

# 并行转换（例如 4 进程）
python convert_checkpoint.py --batch-root experiments/uspto_50k --workers 4

# 覆盖已存在 *_safe.pt 文件
python convert_checkpoint.py --batch-root experiments/uspto_50k --overwrite

# 转换并逐个验证可用性（建议在首次批量迁移时使用）
python convert_checkpoint.py --batch-root experiments/uspto_50k --verify --workers 4
```

可选参数：

- `--dry-run` 只列出计划，不执行
- `--overwrite` 若已存在 `_safe.pt` 仍重新生成
- `--workers N` 进程并行数（>1 使用 `ProcessPoolExecutor`）
- `--verify` 每个输出再用 `torch.load(..., weights_only=True)` 校验结构完整

单文件模式保持不变：

```bash
python convert_checkpoint.py \
  --input experiments/uspto_50k/without_rxn_class/.../epoch_120.pt \
  --output experiments/uspto_50k/without_rxn_class/.../epoch_120_safe.pt \
  --verify
```

Streamlit 选择页已使用该批量接口按钮进行统一转换。

---

## Exporting Predictions

After a successful prediction:

- `Download CSV` → Raw table of Reactant 1 / Reactant 2 / Confidence
- `Download SDF` → Each reactant as a separate record (`$$$$` delimited)

Missing reactant slots (e.g. single-reactant predictions) are skipped.

---

## Configuration

Example YAML:

```yaml
model:
  type: transformer
  d_model: 512
  n_layers: 8
  dropout: 0.1
training:
  epochs: 50
  batch_size: 128
  lr: 1.0e-4
  optimizer: adam
data:
  train: data/processed/train.csv
  val: data/processed/val.csv
inference:
  beam_size: 10
  max_len: 256
device: cuda
seed: 42
```

---

## Logging & Reproducibility

- Set seeds (`random`, `numpy`, `torch`).
- Log per-epoch metrics.
- Save best + final checkpoints.
- Capture config & environment (Python, dependency versions).

---

## Performance & Evaluation

| Metric | Value | Notes |
|--------|-------|-------|
| Top-1 Accuracy | TBD | Validation set |
| Top-5 Accuracy | TBD |  |
| Inference Latency | TBD | Per molecule |
| Model Size | TBD | Checkpoint size |

---

## Deployment

### Streamlit Cloud

Live: <https://smiles-reactor-juwenhao.streamlit.app>

Steps:

1. Push repository to GitHub.  
2. Select app entry `home.py`.  
3. Ensure `requirements.txt` includes RDKit + PyTorch.  
4. Upload / mount checkpoints (or convert online).  

### Local Production

```bash
nohup streamlit run home.py --server.port 8501 --server.address 0.0.0.0 &
```

(Optionally) put Nginx reverse proxy & TLS in front.

---

## Roadmap

- Multi-step route planning
- Diverse decoding (nucleus sampling, diverse beam)
- Confidence calibration
- Batch input file support
- Docker image
- Unit / integration tests
- Ensemble model support

---

## Limitations

- Single-step only (no full synthesis planning).
- No chemical condition feasibility checks.
- Data-dependent generalization.
- Assumes valid SMILES input.

---

## Contributing

```bash
git checkout -b feat/your-feature
# ... work ...
git commit -m "feat: add new decoding strategy"
git push origin feat/your-feature
```

Guidelines:

- Type hints
- Lint (`ruff` / `flake8`)
- Format (`black`)
- Tests for critical logic
- Clear docstrings

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

---

## Acknowledgements

- RDKit
- PyTorch
- Streamlit
- Open-source cheminformatics & ML communities

---

## Citation

```bibtex
@software{ju2025smilesgui,
  author       = {Ju, Wenhao},
  title        = {SMILES-GUI: A Streamlit Interface for Retrosynthesis Prediction},
  year         = {2025},
  url          = {https://github.com/Ju-Wenhao/SMILES-GUI}
}
```

---

## Contact

- Author: Wenhao Ju
- Demo: <https://smiles-reactor-juwenhao.streamlit.app>
- Repo: <https://github.com/Ju-Wenhao/SMILES-GUI>
- Issues: GitHub Issue tracker

---