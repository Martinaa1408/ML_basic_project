# ML_basic_project – Yeast Protein Localization with Machine Learning

This project implements a complete supervised learning pipeline for predicting the **subcellular localization of yeast proteins** using physicochemical descriptors. The solution is built following the principles and assignments of the AML-BASIC course (Applied Machine Learning, University of Bologna, 2025), and emphasizes **metric fairness**, **model robustness**, and **class imbalance handling**.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Pipeline Summary](#pipeline-summary)
- [Results Overview](#results-overview)
- [Model Evaluation](#model-evaluation)
- [Structure](#structure)
- [Models](#models)
- [Reproducibility](#reproducibility)
- [Author](#author)
- [Acknowledgments](#acknowledgments)
- [License](#license)

---

## Project Overview

- **Goal**: Classify yeast proteins into 10 subcellular compartments
- **Type**: Multiclass supervised classification
- **Focus**: Generalization, interpretability, and handling of imbalanced data
- **Tools**: scikit-learn, imbalanced-learn, SMOTE, GridSearchCV, ROC/PR analysis

---

## Dataset

- **Source**: [UCI Yeast Dataset](https://archive.ics.uci.edu/ml/datasets/Yeast)
- **Samples**: 1,484 protein sequences
- **Features**: 8 numeric attributes (e.g., hydrophobicity, signal peptides)
- **Classes**: 10 protein localizations (e.g., CYT, MIT, NUC, POX, ERL)

Note: Two features (`pox`, `erl`) were removed due to >99.99% identical values.

---

## 🔁 Pipeline Summary

| Phase             | Description |
|------------------|-------------|
| **EDA**          | Class distribution, correlation, outlier detection |
| **Preprocessing**| Feature scaling (`StandardScaler`), train/test split, SMOTE |
| **Model Tuning** | GridSearchCV for `C` in SVM and `k` in k-NN |
| **Modeling**     | Logistic Regression, Random Forest, SVM, k-NN |
| **Evaluation**   | Accuracy, Macro-F1, MCC, Confusion Matrix, ROC-AUC, PR-AUC |
| **Export**       | Full project structure, trained models, plots, and scripts |

---

## 📊 Results Overview

| Model               | Accuracy | Macro F1 | MCC     |
|--------------------|----------|----------|---------|
| Logistic Regression| 0.5034    | 0.4960     | 0.3973    |
| **Random Forest**   | **0.6267** | **0.5740** | **0.5170** |
| SVM (C=10)          | 0.5822     | 0.5338     | 0.4640    |
| k-NN (k=5)          | 0.4795    | 0.4583     | 0.3508    |


- Class 2 and 4 showed strong AUC and precision-recall performance.
- Class 7 consistently underperformed due to low representation and overlap.

---

## Model Evaluation

- **Metrics**: Macro-F1, MCC, ROC-AUC, Average Precision
- **Curves**: Multi-class ROC and PR curves (One-vs-Rest)
- **Confusion Matrix**: Biological confusion (e.g., MIT↔CYT) interpreted
- **SMOTE**: Applied adaptively with dynamic `k_neighbors` for minority class safety

---

## Structure

```bash
ML_basic_project/
│
├── data/         # Raw and processed datasets (.csv, .pkl, splits)
├── models/       # Trained models (logreg, rf, svm, knn, best)
├── notebooks/    # Main Jupyter notebook (AML_notebook.ipynb)
├── results/      # Evaluation outputs: plots, confusion matrices, AUC curves
├── scripts/      # Preprocessing utilities (scaling, SMOTE, binarization)
├── report/       # Optional report or PDF submission
├── LICENSE.md    # Custom CC-BY 4.0 license
├── requirements.txt
└── README.md
