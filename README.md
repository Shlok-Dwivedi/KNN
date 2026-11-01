# Lab 5: K-Nearest Neighbors on the Diabetes Dataset

Binary classification (diabetes / no diabetes) using a K-Nearest Neighbors classifier from Scikit-learn, including K selection with cross-validation and hyperparameter tuning with GridSearchCV.

## Overview

**Dataset:** `diabetes.csv` has 768 patients, 8 medical features (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age) and a binary `Outcome` (1 = diabetes, 0 = no diabetes). There are no null values or duplicates, but Glucose, BloodPressure, SkinThickness, Insulin and BMI contain invalid zeros (5, 35, 227, 374 and 11 respectively).

## Workflow

1. Load the data and inspect it (`head`, `tail`, `shape`, `describe`, nulls, duplicates)
2. EDA: Outcome distribution, Age boxplot, correlation heatmap, feature histograms
3. Replace invalid zeros with NaN and fill them with the column median
4. 80:20 stratified train-test split (614 train, 154 test)
5. Standardise features with `StandardScaler` (fitted on training data only)
6. Train KNN with K = 5, predict classes and probabilities
7. Evaluate with accuracy, recall, F1-score and a confusion matrix
8. Compare training and testing accuracy for K = 1 to 30
9. Choose K with stratified 10-fold cross-validation
10. Tune `n_neighbors`, `weights` and `metric` with `GridSearchCV`

## Results (from the notebook run)

| Setting | Result |
|---------|--------|
| KNN, K = 5 (test set) | Accuracy 0.7532, Recall 0.6111, F1 0.6346 |
| Confusion matrix (K = 5) | `[[83, 17], [21, 33]]` |
| Best K on test accuracy | K = 8 (accuracy 0.7662) |
| Best K, 10-fold CV | K = 24 (CV accuracy 0.7770) |
| GridSearchCV (5-fold) | euclidean, K = 19, uniform weights (CV accuracy 0.7769) |

At K = 1 the training accuracy is 1.0, which shows overfitting. Recall for diabetic patients is the weak point: 21 of the 54 diabetic patients in the test set are missed.

## Project structure

```
lab5-knn-diabetes/
├── ML_Lab_5_17_sep_2026.ipynb   # notebook
├── diabetes.csv                 # dataset (add this yourself)
├── requirements.txt
└── README.md
```

## How to run

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook ML_Lab_5_17_sep_2026.ipynb
```

Place `diabetes.csv` in the same folder as the notebook, then run all cells.

## Notes

- The notebook imports `precision_score`, `roc_curve` and `roc_auc_score`, but they are not used. Only accuracy, recall, F1 and the confusion matrix are calculated.
- Choosing K from the test accuracy (K = 8) is less reliable than cross-validation, because the test set has only 154 samples.
