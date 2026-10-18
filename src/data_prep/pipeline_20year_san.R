# ============================================================
# SAN 20-Year Weather Data Pipeline (CORRECTED)
# PCA is fit on training years only (2006-2021)
# ============================================================

library(dplyr)
library(readr)
library(lubridate)
library(tidyr)
library(zoo)

# Task 1: Data Integration
file_paths <- c("SAN_2006_2010.csv", "SAN_2011_2015.csv",
                "SAN_2016_2020.csv", "SAN_2021_2025.csv")

raw_data <- lapply(file_paths, function(file) {
  read_csv(file, comment = "#", na = c("M", "", "NA"), show_col_types = FALSE)
}) %>% bind_rows()

# Task 2: Hourly aggregation
df_hourly <- raw_data %>%
  mutate(
    valid_time = as_datetime(valid),
    hourly_timestamp = round_date(valid_time, unit = "hour")
  ) %>%
  mutate(
    tmpf = suppressWarnings(as.numeric(tmpf)),
    relh = suppressWarnings(as.numeric(relh)),
    sknt = suppressWarnings(as.numeric(sknt)),
    drct = suppressWarnings(as.numeric(drct)),
    vsby = suppressWarnings(as.numeric(vsby)),
    mslp = suppressWarnings(as.numeric(mslp))
  ) %>%
  group_by(hourly_timestamp) %>%
  summarise(
    tmpf = mean(tmpf, na.rm = TRUE),
    relh = mean(relh, na.rm = TRUE),
    sknt = mean(sknt, na.rm = TRUE),
    drct = mean(drct, na.rm = TRUE),
    vsby = mean(vsby, na.rm = TRUE),
    mslp = mean(mslp, na.rm = TRUE),
    .groups = "drop"
  ) %>% ungroup()

# Task 3: Grid + Imputation
full_grid <- tibble(
  hourly_timestamp = seq(
    from = as_datetime("2006-01-01 00:00:00"),
    to = as_datetime("2025-12-31 23:00:00"),
    by = "1 hour"
  )
)

df_complete <- full_grid %>% left_join(df_hourly, by = "hourly_timestamp")

numeric_cols <- c("tmpf", "relh", "sknt", "drct", "vsby", "mslp")

df_imputed <- df_complete %>%
  fill(all_of(numeric_cols), .direction = "down") %>%
  fill(all_of(numeric_cols), .direction = "up")

# Task 4: Unit Conversion
df_transformed <- df_imputed %>%
  mutate(
    date_time = format(hourly_timestamp, "%Y-%m-%d %H:%M:%S"),
    temperature_c         = round((tmpf - 32) * (5 / 9), 1),
    relative_humidity_pct = round(pmin(pmax(relh, 0), 100), 1),
    wind_speed_ms         = round(pmax(sknt, 0) * 0.514444, 1),
    wind_dir_deg          = round(drct),
    visibility_km         = round(pmax(vsby, 0) * 1.60934, 1),
    pressure_hpa          = round(mslp, 1)
  )

# Task 5: Feature Engineering
df_engineered <- df_transformed %>%
  mutate(
    year = year(hourly_timestamp),
    hour_num = hour(hourly_timestamp),
    hour_sin = round(sin(2 * pi * hour_num / 24), 4),
    hour_cos = round(cos(2 * pi * hour_num / 24), 4),
    day_of_year = yday(hourly_timestamp),
    day_of_year_sin = round(sin(2 * pi * day_of_year / 365.25), 4),
    day_of_year_cos = round(cos(2 * pi * day_of_year / 365.25), 4),
    month_num = month(hourly_timestamp),
    month_sin = round(sin(2 * pi * month_num / 12), 4),
    month_cos = round(cos(2 * pi * month_num / 12), 4),
    wind_rad = wind_dir_deg * (pi / 180),
    wind_dir_sin = round(sin(wind_rad), 4),
    wind_dir_cos = round(cos(wind_rad), 4),
    heat_index_c = round(
      temperature_c +
        0.33 * (relative_humidity_pct / 100 * 6.105 * exp((17.27 * temperature_c) / (237.7 + temperature_c))) -
        0.70 * wind_speed_ms - 4.0, 1
    ),
    temp_lag1 = lag(temperature_c, 1, default = first(temperature_c)),
    pressure_lag1 = lag(pressure_hpa, 1, default = first(pressure_hpa)),
    humidity_lag1 = lag(relative_humidity_pct, 1, default = first(relative_humidity_pct)),
    wind_lag1 = lag(wind_speed_ms, 1, default = first(wind_speed_ms)),
    temp_change = round(temperature_c - temp_lag1, 2),
    pressure_change = round(pressure_hpa - pressure_lag1, 2),
    humidity_change = round(relative_humidity_pct - humidity_lag1, 2),
    wind_change = round(wind_speed_ms - wind_lag1, 2),
    wind_chill_c = round(13.12 + 0.6215 * temperature_c -
                           11.37 * (wind_speed_ms * 3.6)^0.16 +
                           0.3965 * temperature_c * (wind_speed_ms * 3.6)^0.16, 1)
  )

df_engineered <- df_engineered %>%
  mutate(
    temp_roll_mean_24h = round(zoo::rollmean(temperature_c, 24, fill = NA, align = "right"), 2),
    temp_roll_sd_24h = round(zoo::rollapply(temperature_c, 24, sd, fill = NA, align = "right"), 2),
    temp_roll_max_24h = round(zoo::rollmax(temperature_c, 24, fill = NA, align = "right"), 2),
    temp_roll_min_24h = round(zoo::rollapply(temperature_c, 24, min, fill = NA, align = "right"), 2),
    humidity_roll_mean_24h = round(zoo::rollmean(relative_humidity_pct, 24, fill = NA, align = "right"), 2),
    wind_roll_mean_24h = round(zoo::rollmean(wind_speed_ms, 24, fill = NA, align = "right"), 2),
    pressure_roll_mean_24h = round(zoo::rollmean(pressure_hpa, 24, fill = NA, align = "right"), 2)
  ) %>%
  mutate(
    temp_range_24h = round(temp_roll_max_24h - temp_roll_min_24h, 2),
    pressure_tendency = round(pressure_hpa - pressure_roll_mean_24h, 2)
  ) %>%
  fill(temp_roll_mean_24h, temp_roll_sd_24h, temp_roll_max_24h,
       temp_roll_min_24h, humidity_roll_mean_24h, wind_roll_mean_24h,
       pressure_roll_mean_24h, temp_range_24h, pressure_tendency,
       .direction = "down") %>%
  fill(temp_roll_mean_24h, temp_roll_sd_24h, temp_roll_max_24h,
       temp_roll_min_24h, humidity_roll_mean_24h, wind_roll_mean_24h,
       pressure_roll_mean_24h, temp_range_24h, pressure_tendency,
       .direction = "up")

# Task 6: PCA fit on TRAINING ONLY (year <= 2021)
pca_input_cols <- c("temperature_c", "relative_humidity_pct", "wind_speed_ms",
                    "pressure_hpa", "visibility_km", "heat_index_c", "wind_chill_c")

train_mask <- df_engineered$year <= 2021

valid_cols <- pca_input_cols[sapply(df_engineered[train_mask, pca_input_cols],
                                    function(x) sd(x, na.rm = TRUE) > 0)]

# Fit PCA on training years only
pca_model <- prcomp(df_engineered[train_mask, valid_cols], center = TRUE, scale. = TRUE)

# Project all rows onto training principal axes
pca_all <- predict(pca_model, newdata = df_engineered[, valid_cols])

df_final <- df_engineered %>%
  mutate(
    PCA1 = round(pca_all[, 1], 3),
    PCA2 = round(pca_all[, 2], 3)
  )

# Task 7: Save (KEEP year column for split)
write_csv(df_final, "SAN_20year_processed_analytics_v2.csv")

drop_cols <- c("hourly_timestamp", "date_time", "hour_num",
               "tmpf", "relh", "sknt", "drct", "vsby", "mslp",
               "day_of_year", "month_num", "wind_rad")

df_model <- df_final %>% select(-any_of(drop_cols))
write_csv(df_model, "SAN_20year_model_ready.csv")

cat("Model-ready features:", ncol(df_model), "\n")
cat("Rows:", nrow(df_model), "\n")
cat("Train rows (<=2021):   ", sum(df_model$year <= 2021), "\n")
cat("Val rows (2022-2023):  ", sum(df_model$year %in% 2022:2023), "\n")
cat("Test rows (>=2024):    ", sum(df_model$year >= 2024), "\n")