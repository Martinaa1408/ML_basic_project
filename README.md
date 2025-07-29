# Predicting Yeast Protein Localization with Machine Learning (AML-BASIC 2025)

[![ML pipeline](https://img.shields.io/badge/GitHub-Run%20full%20ML%20pipeline-grey?logo=github)](https://github.com/Martinaa1408/ML_basic_project/)
[![Notebook](https://img.shields.io/badge/Notebook-ipynb-brightgreen?logo=Jupyter)](notebooks/AML_notebook.ipynb)
[![Scripts](https://img.shields.io/badge/Scripts-Python-blue?logo=python)](scripts/)
[![License](https://img.shields.io/badge/License-CC--BY--NC--SA--4.0-orange?logo=creativecommons)](LICENSE.md)
[![Dataset](https://img.shields.io/badge/Dataset-Yeast-orange?logo=databricks)](https://archive.ics.uci.edu/ml/datasets/Yeast)
[![Course](https://img.shields.io/badge/AML--BASIC-2025-informational?logo=book)](https://drive.google.com/drive/folders/1ZrQpF_F9E45yQTO9mG8Izr3LaECVH0aH)
[![Reproducible](https://img.shields.io/badge/Reproducible-Yes-brightgreen)](requirements.txt)

<img width="517" height="352" alt="image" src="https://github.com/user-attachments/assets/bca22c0f-1141-4287-ad0d-7b9164722fe2" />
---

## Project Overview

This project builds a full machine learning pipeline to classify **yeast proteins** into **10 subcellular compartments** based on numeric sequence features.

Developed as part of the **AML-BASIC 2025 course** at the University of Bologna, it emphasizes:

- Robust generalization under severe class imbalance  
- Fair evaluation with metrics like **Macro-F1** and **MCC**  
- Full pipeline reproducibility with modular code and frozen dependencies  

---

## Problem Framing & Pipeline Design

This is a **multiclass classification task**, where the input is a vector of numeric protein descriptors and the output is one of 10 subcellular compartments.

The pipeline includes:
- Feature filtering and scaling
- SMOTE oversampling with dynamic `k_neighbors`
- Hyperparameter tuning with `GridSearchCV`
- Evaluation with metrics robust to imbalance: **Macro-F1**, **MCC**, **PR-AUC**

---

## Dataset

- **Source**: [UCI Yeast Dataset](https://archive.ics.uci.edu/ml/datasets/Yeast)  
- **Instances**: 1,484 yeast proteins  
- **Classes**: 10 locations (CYT, MIT, NUC, POX, VAC, ME1–ME3, ERL, EXC)  
- **Features**: 8 numeric descriptors → after removing `pox`, `erl` due to low variance  

---

## Pipeline Summary

| Step              | Description |
|------------------|-------------|
| **EDA**           | Correlations, outliers, imbalance analysis |
| **Preprocessing** | `StandardScaler`, drop low-variance features, SMOTE, split |
| **Modeling**      | Logistic Regression, Random Forest, SVM (C=5), k-NN (k=5) |
| **Tuning**        | GridSearchCV for `C` and `k` |
| **Evaluation**    | Confusion Matrix, Macro-F1, MCC, ROC-AUC, PR-AUC |
| **Export**        | Models, plots, processed data, summaries |

---

## Model Overview

| Model               | Accuracy | Macro-F1 |   MCC   |
|--------------------|----------|----------|---------|
| Logistic Regression| 0.8923   | 0.7870   | 0.8642  |
| **Random Forest**   | **0.9832** | **0.9412** | **0.9785** |
| SVM (C=5)          | 0.9832   | 0.9301   | 0.9783  |
| k-NN (k=5)         | 0.8182   | 0.6302   | 0.7712  |

**Observations**:
- RF achieved top test metrics across the board  
- SVM competitive and more stable across folds  
- k-NN best during CV but weak on generalization  

---

## Evaluation & Metrics

- **Macro-F1**: class-wise balanced F1 average  
- **Matthews Correlation Coefficient (MCC)**: balanced multiclass correlation  
- **ROC-AUC** (OvR) and **PR-AUC** curves  
- **Confusion Matrices**: clear CYT↔MIT and POX↔NUC confusions  
- **SMOTE**: dynamically adjusted `k_neighbors` for minority classes  

---

## Why These Models?

- **Logistic Regression**: interpretable baseline, low complexity  
- **Random Forest**: robust, handles non-linearity, low variance  
- **SVM**: strong generalization, margin maximization  
- **k-NN**: intuitive but sensitive to scale and `k`  

---

## Project Structure

```bash
ML_basic_project/
├── data/               # Raw and processed data (.csv, .pkl)
├── models/             # Trained models and GridSearchCV (.pkl)
├── notebooks/          # Jupyter Notebook (AML_notebook.ipynb)
├── results/            # Confusion matrices, ROC/PR curves, tables, performance summary
├── scripts/            # Preprocessing
├── report/             # Final report (PDF + .tex)
├── LICENSE.md          # Custom license (CC BY-NC-SA 4.0)
├── requirements.txt    # Exact dependency versions
└── README.md           # Project overview (you’re here)
```
---

## Reproducibility Checklist

This project ensures **full reproducibility** across data splits, models, metrics, and environment.

### Datasets
- Raw dataset available in `/data/raw/`
- Preprocessed features saved as `.csv` and `.pkl`
- Stratified train/test split stored using fixed `random_state=42`

### Models
- All trained models saved as `.pkl` (`LogisticRegression`, `RandomForest`, `SVM`, `k-NN`)
- Best `GridSearchCV` objects persisted and exportable
- Full hyperparameter grids and validation scores logged

### Metrics & Visualizations
- All evaluation metrics stored:  
  `accuracy`, `macro-F1`, `MCC`, `ROC-AUC`, `PR-AUC`
- Confusion matrices and curves exported as both `.png` and `.pdf`
- Class-level results for PR/ROC saved for auditability

### Pipeline
- Modular Python scripts:
  - Feature filtering
  - Standardization
  - SMOTE with dynamic `k_neighbors` logic
- Deterministic behavior ensured by setting seeds in:
  - `numpy`
  - `scikit-learn`
  - `imbalanced-learn`

### Environment
- All dependencies frozen in `requirements.txt`
- Fully compatible with **Python 3.10+**
- Works on local machines and Google Colab without modification

---

## References

### Dataset & Problem Domain
- Horton, P., & Nakai, K. (1996). *A Probabilistic Classification System for Predicting the Cellular Localization Sites of Proteins*. ISMB. [PubMed](https://pubmed.ncbi.nlm.nih.gov/8877510)

### Machine Learning Models

- **Logistic Regression**  
  - [scikit-learn docs](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)

- **Random Forest**  
  - Breiman, L. (2001). *Random Forests*. [PDF](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf)

- **Support Vector Machine (SVM)**  
  - Cortes, C., & Vapnik, V. (1995). *Support-Vector Networks*. [DOI](https://doi.org/10.1007/BF00994018)

- **k-NN**  
  - [scikit-learn docs](https://scikit-learn.org/stable/modules/neighbors.html)

### Evaluation Metrics

- **Matthews Correlation Coefficient (MCC)** – [scikit-learn MCC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html)  
- **Macro-F1 Score** – [sklearn docs](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)  
- **ROC & PR Curves** – Davis, J., & Goadrich, M. (2006). [ICML Paper](https://dl.acm.org/doi/10.1145/1143844.1143874)

### Class Imbalance

- **SMOTE**  
  - Chawla et al. (2002). *SMOTE: Synthetic Minority Over-sampling Technique*. [DOI](https://doi.org/10.1613/jair.953)  
  - [imbalanced-learn docs](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)

---

## Author

Martina Castellucci  
AML-BASIC 2025 – University of Bologna  
📧 martina.castellucci@studio.unibo.it

---

## License

This project is released under a [Creative Commons BY-NC-SA 4.0 License](LICENSE.md).
