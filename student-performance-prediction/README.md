# Student Performance Prediction (Regression)

This is my portfolio project where I build and compare machine learning regression models to predict a student's final exam grade ($G3$, on a 0–20 scale) based on demographic, social, and academic attributes.

The goal of this project is to explore a real-world educational dataset, build a robust preprocessing pipeline, train regression algorithms, and find out which factors affect student success the most.

## Project Structure
```
student-performance-prediction/
│
├── data/
│   ├── student-mat.csv      # Mathematics course dataset (395 records)
│   ├── student-por.csv      # Portuguese course dataset (649 records)
│   └── student.txt          # Attribute descriptions
│
├── notebook.ipynb           # Executed Jupyter Notebook with results
└── README.md                # Project documentation (this file)
```

---

## Dataset
I used the **Student Performance Dataset** from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/320/student+performance). The data includes student survey responses for a Mathematics course at two Portuguese schools.

Key features include:
* **Academic performance:** `G1` (first period grade), `G2` (second period grade), `failures` (past class failures), `absences`.
* **Study habits:** `studytime` (weekly study time), `schoolsup` (extra support), `famsup` (family support), `activities`.
* **Social and Demographic features:** `age`, `sex`, `address` (urban/rural), `famrel` (family relationship quality), `Dalc`/`Walc` (alcohol consumption).
* **Target:** `G3` (final grade, scale from 0 to 20).

---

## Model Evaluation & Results
I trained and evaluated three regression algorithms on an **80/20 train-test split**:
1. **Linear Regression** (Baseline model)
2. **Decision Tree Regressor** (Tree-based model, limited to `max_depth=5` to prevent overfitting)
3. **Random Forest Regressor** (Ensemble of 100 trees)

### Performance Metrics

| Model | Train MAE | Test MAE | Train RMSE | Test RMSE | Train $R^2$ | Test $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression** | 1.164 | 1.647 | 1.680 | 2.378 | 0.866 | 0.724 |
| **Decision Tree** | 0.515 | 1.357 | 0.845 | 2.467 | 0.966 | 0.703 |
| **Random Forest** | 0.352 | 1.179 | 0.588 | **1.958** | 0.984 | **0.813** |

### My Key Findings
* **Random Forest** performed the best, explaining **81.3% of the variance** in final grades ($R^2 = 0.813$) on the test set, with a Root Mean Squared Error (RMSE) of **1.958 points** on a 20-point scale.
* **Prior Performance is the Strongest Predictor:** `G2` and `G1` (mid-term grades) are highly correlated with the final score. 
* **Behavioral Factors:** When mid-term grades are excluded from the analysis, school **absences** and the number of **past class failures** emerge as the most critical factors predicting student grades.

---

## Predictions for Sample Students
I tested my trained **Random Forest Pipeline** on two hypothetical student profiles:

1. **Student A (High Achiever):** Studies $>10$ hours/week, 0 past failures, 2 absences, parents are teachers, high family relationship quality, $G1 = 16$, $G2 = 17$.
   * **Predicted Final Grade ($G3$): 18.40 / 20**
2. **Student B (Struggling Student):** Studies $<2$ hours/week, 2 past failures, 25 absences, high weekend alcohol consumption, rural address, $G1 = 6$, $G2 = 5$.
   * **Predicted Final Grade ($G3$): 5.86 / 20**

---

## How to Set Up and Run

### 1. Requirements
Install the required scientific packages:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### 2. Run the Notebook
Open the notebook in Jupyter:
```bash
jupyter notebook
```
Navigate to and open `notebook.ipynb`. Since all cell outputs and plots are pre-saved, you can inspect the entire analysis and model comparison immediately without re-running the code.
