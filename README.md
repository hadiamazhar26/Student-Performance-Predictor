# Student Performance Prediction (Regression Analysis)

This is my portfolio project where I build, compare, and analyze machine learning regression models to predict a student's final exam grade ($G3$, on a 0–20 scale) in Mathematics. 

Instead of treating education as a black box, this project aims to understand *which* academic, social, and demographic factors play the biggest role in a student's final results.

---

## 1. Project Directory Layout
```
student-performance-prediction/
│
├── data/
│   ├── student-mat.csv      # Mathematics course dataset (395 records)
│   ├── student-por.csv      # Portuguese language course dataset (649 records)
│   └── student.txt          # Raw dataset documentation and column definitions
│
├── notebook.ipynb           # Executed Jupyter Notebook with outputs, plots, and analysis
└── README.md                # Detailed project documentation
```

---

## 2. Dataset Deep-Dive & Feature Meanings
The dataset represents survey responses and grades from secondary school students in two Portuguese schools: Gabriel Pereira (GP) and Mousinho da Silveira (MS). 

### Target Variable
* **`G3` (Final Grade):** The student's grade in the final period (numeric: from 0 to 20). This is our target variable for regression.

### Demographic & Background Features
1. **`school`:** Student's school (`GP` - Gabriel Pereira, `MS` - Mousinho da Silveira).
2. **`sex`:** Student's sex (`F` - female, `M` - male).
3. **`age`:** Student's age (numeric: 15 to 22).
4. **`address`:** Home address type (`U` - urban, `R` - rural).
5. **`famsize`:** Family size (`LE3` - less/equal to 3, `GT3` - greater than 3).
6. **`Pstatus`:** Parent's cohabitation status (`T` - living together, `A` - apart).
7. **`Medu` / `Fedu`:** Mother's and Father's education level (0 - none, 1 - primary (4th grade), 2 - 5th to 9th grade, 3 - secondary, 4 - higher education).
8. **`Mjob` / `Fjob`:** Mother's and Father's job (`teacher`, `health`, civil `services`, `at_home`, or `other`).
9. **`reason`:** Reason to choose this school (`home` proximity, school `reputation`, `course` preference, or `other`).
10. **`guardian`:** Student's guardian (`mother`, `father`, or `other`).

### Academic & Support Features
11. **`traveltime`:** Home-to-school travel time (1: <15 min, 2: 15–30 min, 3: 30 min to 1 hour, 4: >1 hour).
12. **`studytime`:** Weekly study time (1: <2 hours, 2: 2–5 hours, 3: 5–10 hours, 4: >10 hours).
13. **`failures`:** Number of past class failures (numeric: $n$ if $1 \le n < 3$, else 4).
14. **`schoolsup`:** Extra educational support from school (yes/no).
15. **`famsup`:** Family educational support (yes/no).
16. **`paid`:** Extra paid classes within the course subject (yes/no).
17. **`activities`:** Extra-curricular activities (yes/no).
18. **`nursery`:** Attended nursery school (yes/no).
19. **`higher`:** Wants to take higher education (yes/no).
20. **`internet`:** Internet access at home (yes/no).

### Social & Lifestyle Features
21. **`romantic`:** In a romantic relationship (yes/no).
22. **`famrel`:** Quality of family relationships (numeric: 1 - very bad to 5 - excellent).
23. **`freetime`:** Free time after school (numeric: 1 - very low to 5 - very high).
24. **`goout`:** Going out with friends (numeric: 1 - very low to 5 - very high).
25. **`Dalc` / `Walc`:** Workday and weekend alcohol consumption (numeric: 1 - very low to 5 - very high).
26. **`health`:** Current health status (numeric: 1 - very bad to 5 - very good).
27. **`absences`:** Number of school absences (numeric: 0 to 93).

### Mid-term Grades
28. **`G1` / `G2`:** First and second period grades (numeric: from 0 to 20).

---

## 3. Data Preprocessing Pipeline
Because machine learning models require numeric inputs and are sensitive to feature scales, I built an automated preprocessing pipeline using Scikit-Learn's `ColumnTransformer` and `Pipeline`:

```
Raw Data (X) ──► ColumnTransformer
                  ├── Numerical Columns ──► Median Imputation ──► StandardScaler
                  └── Categorical Columns ──► Mode Imputation ──► OneHotEncoder (drop='first')
```

### Key Preprocessing Steps:
1. **Imputation:**
   * **Numerical Columns:** Handled missing values using `SimpleImputer(strategy='median')`. Median is robust to outliers (e.g., extreme values in `absences`).
   * **Categorical Columns:** Imputed missing values with `SimpleImputer(strategy='most_frequent')` (mode).
2. **One-Hot Encoding:**
   * Transformed categorical variables (e.g. `Mjob`, `reason`) into binary indicator columns.
   * I set `drop='first'` to avoid multicollinearity (the dummy variable trap), which helps linear models remain stable.
3. **Standardization (Feature Scaling):**
   * Applied `StandardScaler` to numerical columns to scale features to have a mean of 0 and standard deviation of 1. This ensures variables with large magnitudes (like `absences`) do not dominate variables with smaller scales (like `studytime`).
4. **Train-Test Split:**
   * I split the preprocessed data into **80% training** and **20% testing** datasets using a fixed seed (`random_state=42`) to guarantee reproducibility.

---

## 4. Models and Evaluation Metrics

I trained and compared three different types of regression algorithms:
1. **Linear Regression:** A simple, interpretable model that fits a linear plane. It serves as my baseline.
2. **Decision Tree Regressor:** A non-parametric model that makes predictions by splitting features. I set `max_depth=5` to prevent the tree from growing too deep and overfitting.
3. **Random Forest Regressor:** An ensemble model that trains 100 independent decision trees (`n_estimators=100`) on random subsets of the data and averages their predictions. This reduces variance and improves generalizability.

### Metric Definitions:
* **MAE (Mean Absolute Error):** The average absolute difference between predicted and actual grades. If MAE is 1.17, it means my predictions are, on average, off by 1.17 points.
* **RMSE (Root Mean Squared Error):** The square root of the average squared errors. Since errors are squared before being averaged, RMSE penalizes larger errors more heavily.
* **$R^2$ Score (Coefficient of Determination):** Represents the proportion of variance in the final grades ($G3$) that the model explains. An $R^2$ of 0.81 means the model explains 81% of the grade differences, while a score of 0.0 would mean it predicts no better than just guessing the mean grade.

---

## 5. Model Comparison Analysis

Here are the evaluation results from the [Jupyter Notebook](./student-performance-prediction/notebook.ipynb):

| Model | Train MAE | Test MAE | Train RMSE | Test RMSE | Train $R^2$ | Test $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression** | 1.164 | 1.647 | 1.680 | 2.378 | 0.866 | 0.724 |
| **Decision Tree** | 0.515 | 1.357 | 0.845 | 2.467 | 0.966 | 0.703 |
| **Random Forest** | 0.352 | 1.179 | 0.588 | **1.958** | 0.984 | **0.813** |

### Key Observations:
* **The Best Performer:** The **Random Forest Regressor** performs best on the test set, achieving the lowest test error (**RMSE = 1.958**) and explaining **81.3% of the variance** ($R^2 = 0.813$).
* **Decision Tree Overfitting:** The single Decision Tree model shows signs of overfitting. It gets a very high $R^2$ of **0.966** on the training set, but drops to **0.703** on the test set. The ensemble Random Forest reduces this variance, boosting test performance.
* **Linear Regression Baseline:** The baseline Linear Regression model performs solidly ($R^2 = 0.724$), showing that many of the relationships in the data are close to linear.

---

## 6. What Factors Actually Predict Student Grades?

Using feature importances extracted from the **Random Forest** model, I analyzed what factors influence student grades.

### Case 1: Including Mid-term Grades (`G1` & `G2`)
When early period grades are included in the dataset, **`G2`** and **`G1`** dominate the importances:
* This indicates that a student's prior academic history in the class is by far the strongest predictor of their final score. If a student does well in mid-terms, they almost always do well in the final.

### Case 2: Excluding Mid-term Grades (`G1` & `G2`)
In a real-world scenario, we might want to predict student success *before* classes start to offer proactive help. When I excluded `G1` and `G2` from the feature pool, the top predictors changed:
1. **`absences`:** The number of absences is the most critical non-grade factor. Missing classes has a direct, negative impact on final exam scores.
2. **`failures`:** The number of historical class failures is highly critical, showing that students with a history of struggling need early intervention.
3. **`health`:** The student's current health status has a significant influence.
4. **`goout`:** How often a student goes out with friends is highly predictive.
5. **`age`:** The student's age (often representing whether they have repeated grades previously).

---

## 7. Pipeline Testing: Mock Student Simulation

To verify that the model makes logical predictions, I created two hypothetical student profiles and passed them through my Random Forest pipeline:

### Student A (High Achiever)
* **Profile:** 17 years old, studies $>10$ hours/week, 0 past failures, 2 absences, parents are teachers, high family relationship quality, $G1=16$, $G2=17$.
* **Pipeline Prediction ($G3$): 18.40 / 20**

### Student B (Struggling Student)
* **Profile:** 18 years old, rural address, studies $<2$ hours/week, 2 past failures, 25 absences, high weekend alcohol consumption, $G1=6$, $G2=5$.
* **Pipeline Prediction ($G3$): 5.86 / 20**

---

## 8. How to Set Up and Run

### 1. Installation
Install the required Python packages:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### 2. Launching the Project
1. Run the Jupyter environment:
   ```bash
   jupyter notebook
   ```
2. Navigate to the `student-performance-prediction/` folder and open **`notebook.ipynb`** to view the code cells and pre-saved outputs. You can select **Cell -> Run All** from the top menu to re-run the entire pipeline.
