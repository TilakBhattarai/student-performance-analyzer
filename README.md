
# Student Performance Analyzer

A Python and NumPy project that analyzes student marks across Math, Science, and English.

## Features

- Loads numerical data from a CSV file using NumPy.
- Calculates subject-wise and student-wise averages.
- Calculates weighted scores using NumPy dot products.
- Identifies high-performing students and the best-performing subject.
- Detects missing values and fills them using subject averages.
- Normalizes scores using z-score normalization and broadcasting.
- Saves normalized results to a CSV file.

## Technologies

- Python
- NumPy
- CSV

## Dataset

The input dataset contains 8 students and 3 subjects, with one intentionally missing mark.

## How to Run

1. Install NumPy:

   ```bash
   pip install numpy
   ```

2. Run the analyzer from the project folder:

   ```bash
   python numpy_analyzer.py
   ```

3. The program prints analysis results and generates a normalized CSV output file.

## Concepts Practiced

NumPy arrays, array properties, axis-based calculations, dot products, boolean masking, missing-value handling, broadcasting, and file input/output.
