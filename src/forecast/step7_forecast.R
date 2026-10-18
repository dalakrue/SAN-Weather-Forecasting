# Thin launcher — the Python file next to this script is canonical.
# The original R file embedded the whole Python script as a string and
# wrote it to a hard-coded Windows path (D:/Users/Lenovo LOQ/...).
# This version just runs the neighbouring .py file, wherever the repo is.
# Run with: Rscript src/forecast/step7_forecast.R  (from the repo root, or any working dir)
args <- commandArgs(trailingOnly = FALSE)
script_arg <- args[grep("--file=", args)]
script_dir <- if (length(script_arg) > 0) {
  dirname(normalizePath(sub("--file=", "", script_arg[1])))
} else {
  getwd()
}
reticulate::py_run_file(file.path(script_dir, "step7_forecast.py"))
