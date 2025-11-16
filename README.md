# Loan Default Prediction

End-to-end data science project using Python, Pandas, and Scikit-learn to predict whether a loan will default based on applicant and loan attributes.

## 1. Problem Statement

Financial institutions need to estimate the risk of loan default before approval.  
The goal of this project is to build a machine learning model that predicts **loan default probability** so lenders can:

- Reduce default risk  
- Design better credit policies  
- Prioritize manual review for high-risk applicants  

## 2. Dataset

- File: `dataset/loan.csv`
- Each row = one loan application
- Target column = **loan default flag** (e.g. `Loan_Status` / `default`)

Key feature types:

- **Applicant demographics** (e.g., gender, marital status, dependents)
- **Financial attributes** (e.g., income, loan amount, loan term)
- **Loan characteristics** (e.g., credit history, property area)

> For privacy reasons, the raw dataset is not uploaded here.  
> The preprocessing and modeling pipeline is fully reproducible with a compatible loan dataset.

## 3. Tech Stack

- **Language:** Python  
- **Libraries:**  
  - Data: `pandas`, `numpy`  
  - Modeling: `scikit-learn`, `imbalanced-learn`  
  - Visualization: `matplotlib`  
  - Other: `pathlib`, `json`

## 4. Approach

1. **Data Loading & EDA**
   - Robust CSV loading with multiple encodings
   - Saved dataset overview to `outputs/data_info.txt`
   - Checked:
     - Shape & dtypes
     - Missing values
     - Sample rows

2. **Target Processing**
   - Automatically detected target column (e.g. `Loan_Status`, `default`, `is_default`)
   - Encoded non-numeric targets (e.g. `Y`/`N`) using `LabelEncoder`
   - Saved label mapping to `outputs/target_label_mapping.json`

3. **Feature Engineering & Preprocessing**
   - Separated features `X` and target `y`
   - Identified:
     - **Numeric features** → imputed missing values (median) + standardized (`StandardScaler`)
     - **Categorical features** → imputed (most frequent) + one-hot encoded (`OneHotEncoder`)
   - Combined into a single `ColumnTransformer` preprocessing pipeline

4. **Train/Test Split**
   - `train_test_split` with:
     - `test_size = 0.2`
     - `random_state = 42`
     - `stratify = y` (to preserve class balance)

5. **Models**
   - **Logistic Regression**
     - Baseline linear model
     - `max_iter = 1000`
   - **Random Forest Classifier**
     - `n_estimators = 200`
     - `class_weight = "balanced"`
     - `random_state = 42`
     - `n_jobs = -1`

6. **Evaluation**
   - Metrics:
     - Accuracy, Precision, Recall, F1-score
     - ROC-AUC
   - Artifacts saved in `outputs/`:
     - Classification reports (JSON)
     - Confusion matrix plots (`*_confusion_matrix.png`)
     - ROC curves (`*_roc_curve.png`)



## 6. How to Run

1. Clone or download this repository
2. Create a virtual environment:

   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate

