# Student Performance Analyzer

A small data pipeline built with NumPy (no loops) that analyzes marks of 8 students across 3 subjects.

## What it does
- Loads marks from `data/scores.csv` (one value is intentionally missing)
- Fills missing values with the subject mean
- Computes subject stats, student averages and z-scores (broadcasting)
- Calculates weighted scores using matrix multiplication (`@`)
- Filters students with average >= 75 and finds the top performer
- Saves results to `results.csv`

## Run
python numpy_analyzer.py

## Concepts used
NaN handling, `axis`, broadcasting, boolean masking, `argmax`, `column_stack`, `savetxt`

## Roadmap
- [x] NumPy version
- [ ] Pandas version (`pandas_analyzer.py`)
