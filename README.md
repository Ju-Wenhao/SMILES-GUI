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

👉 [https://smiles-reactor-juwenhao.streamlit.app](https://smiles-reactor-juwenhao.streamlit.app)

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

User Input (SMILES) → Validation (RDKit) → Tokenization / Encoding → Model Inference (PyTorch) → Candidate Decoding (beam / top-k) → Visualization & Ranking → GUI Output

Conceptual modules:
- Data pipeline (loading, normalization)
- Model (sequence-to-sequence / transformer variant)
- Inference utilities (sampling, beam search)
- Visualization (RDKit drawing)
- Frontend (`home.py` Streamlit app)

---

## Project Structure

(Adjust if your actual repository layout differs.)

```
SMILES-GUI/
├─ home.py                  # Streamlit GUI entry
├─ train.py                 # Training script
├─ requirements.txt
├─ src/
│  ├─ data/                 # Data loading / preprocessing utilities
│  ├─ models/               # Model architectures
│  ├─ inference/            # Decoding / beam search helpers
│  ├─ utils/                # Generic helpers (logging, seeding, config)
│  └─ visualization/        # RDKit-based rendering helpers
├─ data/
│  ├─ raw/                  # Original dataset files
│  ├─ processed/            # Preprocessed artifacts
│  └─ samples/              # Sample subsets
├─ experiments/
│  ├─ checkpoints/          # Trained model weights (*.pt / *.pth)
│  ├─ logs/                 # Training logs / metrics
│  └─ configs/              # (Optional) YAML/JSON configs
├─ docs/
│  ├─ images/               # Screenshots, diagrams
│  └─ design.md
└─ LICENSE
```

---

## Installation

### Prerequisites
- Python 3.9.6
- (Recommended) Conda (especially for RDKit installation)
- (Optional) CUDA-capable GPU for faster training/inference

### Create Environment

```bash
conda create -n smiles-gui python=3.9.6
conda activate smiles-gui
conda install -c conda-forge rdkit
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

Typical dependencies (subset):
- torch
- rdkit (or rdkit-pypi)
- streamlit
- numpy
- pandas
- tqdm
- (optional) pyyaml, scikit-learn

---

## Quick Start

1. Ensure at least one model checkpoint exists under `experiments/checkpoints/`.
2. Run the GUI locally:
   ```bash
   streamlit run home.py
   ```
3. Enter a product SMILES (e.g. `CCOC(=O)C1=CC=CC=C1`).
4. Click the predict button to view reactant candidates.

---

## Usage (GUI Inference)

Launch:
```bash
streamlit run home.py
```

Typical UI elements (may vary):
- Text input: target product SMILES
- (Optional) model selector
- (Optional) decoding parameters: beam size / top-k
- Predict trigger button
- Output panel: ranked reactant sets + molecule images

Feature ideas you can extend:
- Downloadable CSV of predictions
- Confidence / score metrics
- Multi-candidate comparison view
- Batch input via uploaded file

---

## Model Training

Use the training script:

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

Suggested arguments (align with your implementation):
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

Checkpoint naming convention example:
```
experiments/checkpoints/model_epoch{E}_val{metric}.pt
```

---

## Data & Checkpoints

| Path | Purpose |
|------|---------|
| `data/raw/` | Original unmodified dataset(s) |
| `data/processed/` | Tokenized / standardized training artifacts |
| `experiments/checkpoints/` | Saved model weights |
| `experiments/logs/` | Training event / metric logs |
| `experiments/configs/` | Configuration files (if used) |

Recommendation:
- Keep large raw datasets out of version control (use `.gitignore`).
- Save a copy of each config used for a training run.

---

## Configuration

If you employ config files (YAML/JSON), an example:

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
  vocab: data/processed/vocab.json
inference:
  beam_size: 10
  max_len: 256
device: cuda
seed: 42
```

---

## Logging & Reproducibility

Recommended practices (ensure implemented as needed):
- Set global seeds (`random`, `numpy`, `torch`).
- Log per-epoch training & validation metrics.
- Save:
  - Best checkpoint (based on validation metric)
  - Final checkpoint
  - Config snapshot
- (Optional) Record Git commit hash.

---

## Performance & Evaluation

Fill in when metrics are available:

| Metric | Value | Notes |
|--------|-------|-------|
| Top-1 Accuracy | TBD | Validation set |
| Top-5 Accuracy | TBD | |
| Inference Latency | TBD | Per molecule |
| Model Size | TBD | Checkpoint file size |

Add reaction class–conditioned metrics if your dataset contains labels.

---

## Deployment

### Streamlit Cloud (Public App)
The live instance runs at:  
https://smiles-reactor-juwenhao.streamlit.app

Typical deployment steps:
1. Push repository to GitHub (public).
2. On Streamlit Cloud, select the repo and specify `home.py` as the entry script.
3. Set Python version to 3.9.x in a `packages` or `runtime.txt` (if needed).
4. Ensure `requirements.txt` includes all dependencies.
5. Upload or reconstruct checkpoint into `experiments/checkpoints/` (avoid committing large files if exceeding limits; consider external hosting).

### Local Production (Example)
Use `nohup` or a process manager:
```bash
nohup streamlit run home.py --server.port 8501 --server.address 0.0.0.0 &
```

Reverse proxy (Nginx) can be added for SSL/hostname.

---

## Roadmap

- Add multi-step planning extension
- Improve decoding diversity (diverse beam / nucleus sampling)
- Confidence scoring calibration
- Dataset curation utilities
- Batch CSV upload via GUI
- Dockerfile & container deployment
- Unit tests for tokenizer and inference functions
- Model ensemble support

---

## Limitations

- Single-step retrosynthesis only (no route planning).
- Does not validate chemical feasibility (reagents, conditions, yields).
- Model quality tied to dataset distribution.
- No integration with external reaction knowledge bases.
- GPU recommended for larger models; CPU inference may be slow.

---

## Contributing

1. Fork the repository
2. Create a branch:
   ```bash
   git checkout -b feature/your-feature
   ```
3. Commit with conventional message:
   ```bash
   git commit -m "feat: add improved beam search"
   ```
4. Push & open a Pull Request

Suggested style:
- Use type hints
- Lint (e.g. `ruff` or `flake8`)
- Format (e.g. `black`)
- Add docstrings (Google / NumPy style)
- Add tests for critical logic

---

## License

Licensed under the Apache License 2.0.  
See the [LICENSE](LICENSE) file for full text.

```
Apache License
Version 2.0, January 2004
```

---

## Acknowledgements

- RDKit
- PyTorch
- Streamlit
- Open-source cheminformatics & ML communities
- Academic supervision and institutional support

---

## Citation

If you use this project:

```
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
- Live Demo: https://smiles-reactor-juwenhao.streamlit.app
- Repository: https://github.com/Ju-Wenhao/SMILES-GUI
- Issues: Please open a GitHub Issue for bug reports or feature requests.

---

Feel free to let me know if you’d like:
- A Chinese translation
- A minimized README for PyPI or Docker Hub
- A badge section (e.g. license, Python version, demo status)