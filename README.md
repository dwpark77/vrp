# VRP Sample Solver

This repository includes a basic solver for the vehicle routing and loading problem.

## Requirements

- Python 3.12
- `ortools`, `pandas`, `openpyxl`, `numpy`

Install dependencies with:

```bash
pip install ortools pandas openpyxl matplotlib
```

## Usage

Run the solver script from the repository root:

```bash
python3 solve.py
```

The script reads the sample dataset in `problem-docs` and creates `Result.xlsx` in the same directory.
