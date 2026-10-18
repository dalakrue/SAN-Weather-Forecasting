library(readr)
library(dplyr)

ind <- read_csv("multi_target_performance_indicators.csv", show_col_types = FALSE)

# ---- Best model per target (by lowest MAE) ----
best_per_target <- ind %>%
  group_by(Target) %>%
  slice_min(MAE, n = 1) %>%
  ungroup() %>%
  select(Target, Best_Model = Model, MAE, RMSE, R2, MAPE_pct)

cat("\n=========== BEST MODEL PER TARGET (by TEST MAE) ===========\n")
print(as.data.frame(best_per_target), row.names = FALSE)

# ---- Overall ranking (average MAE across all targets) ----
overall <- ind %>%
  group_by(Model) %>%
  summarise(
    Avg_MAE  = round(mean(MAE), 3),
    Avg_RMSE = round(mean(RMSE), 3),
    Avg_R2   = round(mean(R2), 4),
    .groups = "drop"
  ) %>%
  arrange(Avg_MAE)

cat("\n=========== OVERALL MODEL RANKING (by Avg MAE) ===========\n")
print(as.data.frame(head(overall, 8)), row.names = FALSE)

cat("\n=========== CONCLUSION ===========\n")
cat("Best overall model:", overall$Model[1],
    "with average MAE of", overall$Avg_MAE[1], "and average R2 of", overall$Avg_R2[1], "\n")

write.csv(best_per_target, "final_best_per_target.csv", row.names = FALSE)
write.csv(overall, "final_overall_ranking.csv", row.names = FALSE)

cat("\nSaved: final_best_per_target.csv, final_overall_ranking.csv\n")