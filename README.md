# Dslr

## Introduction
DSLR is an intermediate project in 42's machine learning curriculum, introducing two key concepts: **logistic regression** and **data visualization**.

We are provided with a dataset containing various characteristics of **1,600 Hogwarts students** (such as their grades in different subjects, birth date, name, etc.), along with their respective Houses. This is a **classification problem**, where the goal is to train a model that can predict a student’s House—one of the four Hogwarts Houses—based on their characteristics.

## Project Overview
The project consists of several steps:
1. **Data Visualization** – Explore and clean the dataset while identifying the most relevant features for training.
2. **Model Training** – Train a logistic regression model using the data from `dataset_train.csv`.
3. **Prediction** – Use the trained model to predict the Houses of students listed in `dataset_test.csv`.

## Technical Specifications
- The entire project is written in **Python 3.6.9**.
- Utilizes the following libraries: **matplotlib, pandas, numpy**.
- Datasets are stored in **CSV files**.
- The project includes **six programs**:
  - `describe.py`
  - `histogram.py`
  - `scatter_plot.py`
  - `pair_plot.py`
  - `logreg_train.py`
  - `logreg_predict.py`
- `logreg_train.py` must be executed before `logreg_predict.py`.

## Program Descriptions
### `describe.py`
Provides an overview of the dataset’s features, similar to `pandas.describe()`.

### `histogram.py`
Generates histograms for each subject, displaying grade distributions in different colors based on students’ Houses.

### `scatter_plot.py`
Creates scatter plots for all possible subject pairings to reveal correlations.

### `pair_plot.py`
Displays a pair plot of student grades, combining histograms and scatter plots.

### `logreg_train.py`
Trains a **logistic regression model** using five selected subjects, chosen based on prior data visualization. Since there are four Houses, a **one-vs-all** approach is used. The resulting model weights are stored in `weights.npy`.

Options:
- `-c`: Calculate the costs.
- `-e`: Set the number of iterations.
- `-lr`: Set the learning rate (between 0 and 1).

### `logreg_predict.py`
Uses the previously computed weights (`weights.npy`) to predict the Houses of students listed in `dataset_test.csv`. The predicted Houses, along with student indices, are saved in `houses.csv`.

## Results
The final model achieves a **98.75% accuracy** compared to the actual Houses listed in `dataset_truth.csv`.
