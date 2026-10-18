# src/data_prep/ — R data pipeline (step 1)

## Files

| File | What it does |
|---|---|
| `pipeline_20year_san.R` | Full data pipeline (was `20-year 2006-2025 weather for San Diego.R`, renamed — spaces in filenames break scripts). |

## How to run

```bash
# From a dir containing SAN_2006_2010.csv, SAN_2011_2015.csv,
# SAN_2016_2020.csv, SAN_2021_2025.csv (see data/README.md):
Rscript src/data_prep/pipeline_20year_san.R
# Writes SAN_20year_processed_analytics_v2.csv + SAN_20year_model_ready.csv
```

R packages: `dplyr`, `readr`, `lubridate`, `tidyr`, `zoo`.

## What it does

1. Merge the 4 source CSVs (2006–2025), parsing `M` / empty as NA.
2. Hourly aggregation (mean per hour).
3. Full hourly grid 2006-01-01 → 2025-12-31 + up/down imputation.
4. Feature engineering: 1-h lags, 24-h rolling mean/sd/max/min,
   wind direction sin/cos, heat index, wind chill, temp/pressure/
   humidity/wind change, temp range + pressure tendency.
5. PCA fit on **training years only** (≤2021), projected onto all rows
   (PCA1, PCA2) — no test leakage.
6. Save model-ready CSV with `year` kept for splitting.

No hard-coded paths — it reads/writes the working directory.
