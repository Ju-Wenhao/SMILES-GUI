# SMILES-GUI

A lightweight Streamlit interface for single-step retrosynthesis (inverse synthesis) from a product SMILES using PyTorch models and RDKit for validation & rendering.

Live Demo: https://smiles-reactor-juwenhao.streamlit.app  
Author: Ju Wenhao

---

## 1. What It Does

Input a product SMILES → validate (RDKit) → run a trained model → propose reactant sets → visualize + export (CSV / SDF).

---

## 2. Key Features

| Area | Summary |
|------|---------|
| Input | Free-form SMILES with RDKit sanitization |
| Inference | Single-step retrosynthesis (reactant set prediction) |
| Decoding | Beam / top-k style (extensible) |
| Visualization | Product + reactants (RDKit) |
| Checkpoints | Safe format + batch conversion utility |
| Training | Stand‑alone script (`train.py`) |
| Export | CSV & SDF downloads |
| Caching | In-session model reuse |
| Deployment | Local or Streamlit Cloud |

---

## 3. Quick Start

```bash
# Environment (recommended)
conda create -n smiles-gui python=3.9.6 -y
conda activate smiles-gui
conda install -c conda-forge rdkit -y
pip install -r requirements.txt

# Launch GUI
streamlit run home.py
```

Open the provided URL. Enter a product SMILES (e.g. `CCOC(=O)C1=CC=CC=C1`) and click Predict.

---

## 4. Training (Basic)

```bash
python train.py \
  --data_dir data/processed \
  --train_file train.csv \
  --val_file val.csv \
  --output_dir experiments/checkpoints/run_001 \
  --epochs 50 --batch_size 128 --lr 1e-4 --device cuda --seed 42
```

Outputs: checkpoints + logs under `experiments/`.

---

## 5. Checkpoints & Conversion

Single conversion:
```bash
python convert_checkpoint.py \
  --input path/to/epoch_120.pt \
  --output path/to/epoch_120_safe.pt \
  --verify
```

Batch (recursive):
```bash
python convert_checkpoint.py --batch-root experiments/uspto_50k --verify --workers 4
```

Useful flags: `--dry-run`, `--overwrite`, `--workers N`, `--verify`.

---

## 6. Exporting Predictions

After inference:  
- CSV: Reactant columns + confidence  
- SDF: Each reactant as a record (`$$$$` delimited)  

---

## 7. Minimal Directory Overview

```
home.py              # Streamlit entry
models/              # Model + decoding
utils/               # Chemistry & graph helpers
data/                # Raw / processed datasets
experiments/         # Checkpoints / logs
convert_checkpoint.py
train.py
requirements.txt
```

---

## 8. Roadmap (Short)

- Multi-step planning
- Diverse decoding (nucleus / diverse beam)
- Batch input file support
- Docker image
- Unit tests
- Ensemble support

---

## 9. Limitations

- Single-step only
- No reaction condition prediction
- Dataset-dependent generalization
- Assumes valid input SMILES

---

## 10. Citation

```bibtex
@software{ju2025smilesgui,
  author = {Ju, Wenhao},
  title  = {SMILES-GUI: A Streamlit Interface for Retrosynthesis Prediction},
  year   = {2025},
  url    = {https://github.com/Ju-Wenhao/SMILES-GUI}
}
```

---

## 11. License

Apache 2.0 (see LICENSE)

---

## 12. Contact

- Demo: https://smiles-reactor-juwenhao.streamlit.app  
- Repo: https://github.com/Ju-Wenhao/SMILES-GUI  
- Issues: GitHub issue tracker

---
