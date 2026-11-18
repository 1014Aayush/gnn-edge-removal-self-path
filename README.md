# Noise to One, Signal to Another

Code for *Noise to One, Signal to Another: Under Strong Heterophily, a Separate Self-Path Decides Whether Edges Help a GNN* (A. Pokhrel, 2026).

## Files
| File | Purpose |
|---|---|
| `edge_removal.py` | Edge-removal strategies (random, heterophilic-first, homophilic-first) |
| `models.py` | GCN, GraphSAGE, GAT, GAT-sep, GCN-res, MLP |
| `run.py` | Sparsity sweep (resumable) |
| `analyze.py`, `analyze_v5.py` | Tables, figures and prediction checks |
| `notebooks/` | Kaggle notebook used for the runs (2× T4) |
| `tests/` | Unit tests (`pytest tests`) |

## Usage
```bash
pip install -r requirements.txt
python run.py --help
python analyze_v5.py <result csvs> --outdir results/v5
```
Datasets (Cora, Roman-empire, Amazon-ratings, Tolokers) are downloaded by PyTorch Geometric.
