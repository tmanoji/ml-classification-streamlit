# Breast Cancer Diagnosis: Comparing Five ML Classifiers with Streamlit

## a. Problem statement

Breast cancer is one of the most common cancers worldwide, and early, accurate diagnosis of a tumour as **malignant** or **benign** is critical for treatment. The goal of this project is to build and compare five machine-learning classification models that predict the diagnosis of a breast mass from measurements of its cell nuclei. The models are evaluated with six metrics and demonstrated through an interactive Streamlit web application deployed on Streamlit Community Cloud.

This is a **binary classification** problem. The positive class (1) is malignant and the negative class (0) is benign.

## b. Dataset description

| Item | Details |
|---|---|
| Name | Breast Cancer Wisconsin (Diagnostic) |
| Source | UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic |
| Instances | 569 |
| Features | 30 numeric, real-valued features |
| Target | `diagnosis`: 1 = malignant (212), 0 = benign (357) |
| Missing values | None |

The features are computed from a digitised image of a fine-needle aspirate (FNA) of a breast mass. They describe the characteristics of the cell nuclei in the image. Ten base measurements are recorded:

- radius
- texture
- perimeter
- area
- smoothness
- compactness
- concavity
- concave points
- symmetry
- fractal dimension

Three statistics are computed for each measurement: the **mean**, the **standard error** and the **worst** (the largest value). This gives 30 features, for example `mean_radius`, `radius_error` and `worst_radius`.

**Preprocessing:**
- The data was split 80/20 into train and test sets, stratified by class (`random_state=42`). This gives 455 training rows and 114 test rows.
- The test split is saved as `test_data.csv`. It has the 30 features plus `diagnosis`.
- The features are standardised with `StandardScaler` inside each model's pipeline. The Decision Tree is the exception, because trees are not affected by feature scale. The scaler is saved together with each model, so the uploaded test data does not need to be scaled.

## c. GitHub repository link

<!-- Replace with your repository URL -->
https://github.com/tmanoji/ml-classification-streamlit

**Live Streamlit app:** https://tmanoji-ml-classifier.streamlit.app

```
project-folder/
├── app.py                  # Streamlit app (loads saved models, evaluates on uploaded CSV)
├── train_models.py         # Trains the 5 models, saves models + metrics + test data
├── requirements.txt
├── README.md
├── test_data.csv           # Held-out test data (30 features + diagnosis)
├── data/
│   └── breast_cancer.csv   # Full dataset (569 rows)
└── model/
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── support_vector_machine.pkl
    ├── feature_columns.pkl # Feature order expected by the models
    └── metrics.csv         # Test-set metrics for all models
```

## d. Models used

The five models below were trained on the same training split and evaluated on the same test set of 114 rows (72 benign, 42 malignant):

| Model | Configuration |
|---|---|
| Logistic Regression | `StandardScaler` → `LogisticRegression(max_iter=1000)` |
| Decision Tree | `DecisionTreeClassifier(max_depth=5)` |
| KNN | `StandardScaler` → `KNeighborsClassifier(n_neighbors=5)` |
| Naive Bayes | `StandardScaler` → `GaussianNB()` |
| Support Vector Machine | `StandardScaler` → `SVC(kernel="rbf")`. AUC is computed from `decision_function` scores. |

### Comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9649 | **0.9960** | 0.9750 | **0.9286** | 0.9512 | 0.9245 |
| Decision Tree | 0.9211 | 0.9448 | 0.9459 | 0.8333 | 0.8861 | 0.8299 |
| KNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9211 | 0.9891 | 0.9231 | 0.8571 | 0.8889 | 0.8292 |
| Support Vector Machine | **0.9737** | 0.9947 | **1.0000** | **0.9286** | **0.9630** | **0.9442** |

Precision, Recall and F1 are calculated for the positive class, malignant (1).

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | A very strong linear baseline. It has the highest AUC (0.996), which means it ranks malignant cases above benign ones almost perfectly. It made 4 errors: 1 false positive and 3 missed malignant cases. This shows the classes are close to linearly separable once the features are standardised. |
| Decision Tree | The weakest model overall: lowest accuracy (0.921), recall (0.833) and AUC (0.945). It missed 7 of 42 malignant tumours, the most of any model, which is a serious problem for a medical diagnosis task. A single tree splits on one feature at a time with hard thresholds, so it has high variance and generalises less well than the other models. A depth limit of 5 was used to reduce overfitting. |
| KNN | Good performance once the features are standardised: accuracy 0.956, precision 0.974. It missed 4 malignant cases. KNN depends on distances, so scaling is essential; without it, large-valued features such as `area` would dominate. It also has to store all training data, which makes prediction slower. |
| Naive Bayes | AUC is high (0.989) but accuracy is lower (0.921). It produced the most false positives (3) and missed 6 malignant cases. The model assumes the features are independent, but many of these features are strongly correlated (for example radius, perimeter and area), so that assumption doesn't hold. Even so, it is fast and gives reasonable probability rankings. |
| Support Vector Machine | The best model: highest accuracy (0.974), F1 (0.963) and MCC (0.944), with perfect precision (no false positives). The RBF kernel learns a flexible, maximum-margin boundary in the standardised 30-dimensional feature space. It still missed 3 malignant cases, the same as Logistic Regression. |
| **Overall winner for this dataset** | **Support Vector Machine (RBF).** It leads on accuracy, F1 and MCC, and MCC is the most reliable single metric when the classes are imbalanced (63% benign, 37% malignant). Logistic Regression is a very close second and has the best AUC. For screening, recall is the most important metric, and both models catch 39 of 42 malignant cases. |

## Streamlit app features

- **CSV upload** of test data. A sample `test_data.csv` can be downloaded from the sidebar, or the bundled copy can be used directly.
- **Model selection dropdown** to choose any of the five saved models.
- **Evaluation metrics** for the selected model: Accuracy, AUC, Precision, Recall, F1 and MCC.
- **Confusion matrix** heatmap and a **classification report**.
- A **comparison table of all five models** on the uploaded data, with the best value in each column highlighted.
- **Downloadable predictions** for the selected model.

The app only loads the saved models from `model/`. It does not retrain anything on the uploaded data.
