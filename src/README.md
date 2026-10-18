# src/ — Pipeline

| Folder | Step | What it does |
|---|---|---|
| `data_prep/` | 1 | R: raw 5-year CSVs → model-ready CSV |
| `features/` | 5 | Python: 24-h sequences + MinMax scaling |
| `models/` | 6 | Python: 8 models + val-weighted ensemble |
| `evaluation/` | 8 | Metrics, rankings, accuracy tables |
| `forecast/` | 7 | Forecast charts |

Run order: `data_prep` → `features` → `models` → `evaluation` → `forecast`.
(Step numbers match the original project: 5, 6, 7, 8.)
Each subfolder has its own README. `.R` files are thin launchers —
the `.py` files are canonical.
