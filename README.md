# Titanic Data Journey

## Purpose
Description: This dataset comes from Kaggle containing information about the passengers on the famous Titanic ship which sank on the 15th of April 1912. For the problem set, we explore the best practises of creating a good python project, how to refactor and how to collaborate using Git and Github. There are four members of the group who will create four seperate branches for each section of work.

## Setup Instructions

To install this project, you first need to fork the repository from: https://github.com/sajid-ahmed1/ps1_ex4_nb_to_repo

Then clone your forked repository locally:
```python
git clone https://github.com/<username>/ps1_ex4_nb_to_repo.git
cd <repo-name>
```

Then create the environment using the provided YAML file in the repository:
```bash
conda env create -f environment.yml
```

Then activate the environment:
```bash
conda activate titanic-env
```

Once done, you can work on the project, commit, push it to your fork and open a pull request for the project owner (sa2357) can review the code and approve it.

## Folder Structure

### File Overview
```css
├── data/ /*Where the data is stored*/
├── notebooks/ /*Where the analysis jupyter notebook is stored*/
├── src/ /*Where the .py functions are refactored and stored*/
├── environment.yml /*Information regarding the environment and packages used to run the project without dependency conflicts*/
└── README.md /*Information regarding the project scope and installation*/
```

### src/ Folder Overview
```css
├── src/
    ├── data.py /*Python function to load the data in*/
    ├── utils.py /*Python functions that contain helper utilities such as spliting the data*/
    ├── visualize.py /*Python functions that use matplotlib and seaborn to visualise the data*/
    ├── model.py /*Python function to train the data on a Random Forest model*/
```

## Meta Information
### Member Information

#### Member 1 (Include your name here) - Data ingestion & cleaning
Files: 
- [] src/data.py
- [] src/utils.py

Tasks:
- [] Create load_data(path) to read CSVs.
- [] Implement cleaning: handle missing values, encode categorical variables, feature selection.
- [] Add helper utilities (train/test split, column renaming).
- [] Write docstrings for each function.

#### Member 2 (Include your name here) — Exploratory data analysis & visualization
Files: 
- [] src/visualize.py
- [] notebooks/analysis.ipynb (EDA section)

Tasks:
- [] Implement plots: survival rates by gender/class, age histograms, correlations.
- [] Use matplotlib or seaborn.
- [] Replace notebook EDA code cells with function calls.
- [] Keep charts consistent and reusable.

#### Member 3 (Include your name here) — Modeling & evaluation
Files: 
- [] src/model.py

Tasks:
- [] Build training and evaluation pipeline (e.g., logistic regression or random forest).
- [] Include train_model(X, y) and evaluate_model(model, X_test, y_test).
- [] Save/load model functionality if needed.
- [] Add docstrings and simple metric outputs (accuracy, precision, recall).

#### Member 4 (sa2357) — Repository structure & environment management
Files: 
- [] environment.yml
- [x] README.md
- [x] project root

Tasks:
- [] Create and maintain environment.yml.
- [] Write clear README.md describing structure and usage.
- [] Ensure folder hierarchy
- [] Manage version control (branch naming, pull requests, code review workflow).

### Documentation and Datasets
Dataset originally from Kaggle: https://www.kaggle.com/datasets/sakshisatre/titanic-dataset
Task assigned by Adrian Ochs for D100: https://open-wound-6e6.notion.site/Ex-4-Jupyter-notebook-to-proper-repo-28f5fbd9d211807893e8dfddf7b3ae42