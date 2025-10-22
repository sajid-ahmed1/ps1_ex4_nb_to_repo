# Titanic Data Journey

## Task Structure

### Member 1 (Include your name here) - Data ingestion & cleaning
Files: 
- [] src/data.py
- [] src/utils.py

Tasks:
- [] Create load_data(path) to read CSVs.
- [] Implement cleaning: handle missing values, encode categorical variables, feature selection.
- [] Add helper utilities (train/test split, column renaming).
- [] Write docstrings for each function.

### Member 2 (Include your name here) — Exploratory data analysis & visualization
Files: 
- [] src/visualize.py
- [] notebooks/analysis.ipynb (EDA section)

Tasks:
- [] Implement plots: survival rates by gender/class, age histograms, correlations.
- [] Use matplotlib or seaborn.
- [] Replace notebook EDA code cells with function calls.
- [] Keep charts consistent and reusable.

### Member 3 (Include your name here) — Modeling & evaluation
Files: 
- [] src/model.py

Tasks:
- [] Build training and evaluation pipeline (e.g., logistic regression or random forest).
- [] Include train_model(X, y) and evaluate_model(model, X_test, y_test).
- [] Save/load model functionality if needed.
- [] Add docstrings and simple metric outputs (accuracy, precision, recall).

### Member 4 (sa2357) — Repository structure & environment management
Files: 
- [x] environment.yml
- [x] README.md
- [x] project root

Tasks:
- [] Create and maintain environment.yml.
- [] Write clear README.md describing structure and usage.
- [] Ensure folder hierarchy
- [] Manage version control (branch naming, pull requests, code review workflow).