# ML_basic_project – Yeast Protein Localization with Machine Learning

[![Run full ML pipeline](https://img.shields.io/badge/GitHub-Run%20full%20ML%20pipeline-grey?logo=github)](https://github.com/) 
[![Notebook](https://img.shields.io/badge/Notebook-ipynb-brightgreen?logo=Jupyter)](notebooks/AML_notebook.ipynb)
[![Scripts](https://img.shields.io/badge/Scripts-Python-blue?logo=python)](scripts/)
[![License](https://img.shields.io/badge/License-CC--BY--NC--SA--4.0-orange?logo=creativecommons)](LICENSE.md)
[![Dataset](https://img.shields.io/badge/Dataset-Yeast-orange?logo=databricks)](https://archive.ics.uci.edu/ml/datasets/Yeast)
[![Course](https://img.shields.io/badge/AML--BASIC-2025-informational?logo=book)](https://drive.google.com/drive/folders/1ZrQpF_F9E45yQTO9mG8Izr3LaECVH0aH)

---

## Project Overview

This project tackles the classification of **yeast proteins** into **10 subcellular compartments** using supervised machine learning. Built as part of the **AML-BASIC 2025 course** at the University of Bologna, the pipeline emphasizes:

- Generalization and robust evaluation
- Reproducibility of results
- Fair metric reporting (especially on imbalanced data)
- Interpretability of model behavior

---

## Theoretical Overview

This is a **supervised learning** problem, specifically a **multiclass classification** task.

- **Input**: Numeric protein descriptors  
- **Output**: One of 10 discrete classes (subcellular locations)  
- **Goal**: Learn a hypothesis \( h: \mathbb{R}^n \rightarrow \{1, ..., 10\} \)

We apply a full machine learning pipeline as outlined in the AML-BASIC lectures, including:

- Feature engineering and scaling
- Dealing with class imbalance using **SMOTE**
- Hyperparameter tuning with **GridSearchCV**
- Bias-variance diagnostics and regularization
- Evaluation using **Macro-F1**, **MCC**, **ROC-AUC**, and **PR-AUC**

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
├── results/            # Confusion matrices, ROC/PR curves, tables
├── scripts/            # Preprocessing, scaling, SMOTE, training utils
├── report/             # Final report or presentation (optional)
├── LICENSE.md          # Custom license (CC BY-NC-SA 4.0)
├── requirements.txt    # Exact dependency versions
└── README.md           # Project overview (you’re here)
```
---

## Reproducibility Checklist

This project was designed to ensure **full reproducibility** across data splits, models, and metrics. Below are the reproducibility guarantees implemented.

- **Datasets**
  - Raw dataset available in `/data/raw/`
  - Preprocessed features saved as `.csv` and `.pkl`
  - Stratified train/val/test splits stored with fixed `random_state`

- **Models**
  - All trained models (`LogisticRegression`, `RandomForest`, `SVM`, `k-NN`) saved as `.pkl`
  - Best `GridSearchCV` objects persisted for analysis
  - Parameter grids and cross-validation scores are logged

- **Metrics & Plots**
  - All performance metrics saved: `accuracy`, `macro-F1`, `MCC`, `ROC-AUC`, `PR-AUC`
  - Confusion matrices saved as both `.png` and `.pdf`
  - ROC and PR curves generated One-vs-Rest for multiclass setting

- **Pipeline**
  - Modular preprocessing scripts:
    - Feature filtering
    - Standardization
    - SMOTE augmentation with safe `k_neighbors`
  - Deterministic random seeds across `numpy`, `sklearn`, and `imblearn`

- **Environment**
  - `requirements.txt` provided with pinned versions
  - Compatible with Python 3.10+
  - Ready to be run on local or Colab environments

---

## References

### Dataset & Problem Domain
- Horton, P., & Nakai, K. (1996). *A Probabilistic Classification System for Predicting the Cellular Localization Sites of Proteins*. ISMB. [PubMed](https://pubmed.ncbi.nlm.nih.gov/8877510)
- UCI Machine Learning Repository – Yeast Dataset: [https://archive.ics.uci.edu/ml/datasets/Yeast](https://archive.ics.uci.edu/ml/datasets/Yeast)

### Machine Learning Models

- **Logistic Regression**  
  - [scikit-learn doc](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
  - A linear model for classification, minimizing cross-entropy via L2 regularization.

- **Random Forest**  
  - [Breiman, 2001](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf): *Random Forests*
  - [scikit-learn doc](https://scikit-learn.org/stable/modules/ensemble.html#random-forests)
  - An ensemble of decision trees using bagging and feature randomness for low-variance predictions.

- **Support Vector Machine (SVM)**  
  - Cortes, C., & Vapnik, V. (1995). *Support-Vector Networks*. Machine Learning. [DOI](https://doi.org/10.1007/BF00994018)
  - [scikit-learn doc](https://scikit-learn.org/stable/modules/svm.html)
  - Maximizes the margin between classes using linear/non-linear kernels.

- **k-Nearest Neighbors (k-NN)**  
  - [scikit-learn doc](https://scikit-learn.org/stable/modules/neighbors.html#nearest-neighbors-classification)
  - Instance-based learner using Euclidean distance; sensitive to scaling and choice of `k`.

### Evaluation Metrics

- **Matthews Correlation Coefficient (MCC)**  
  - [Wikipedia](https://en.wikipedia.org/wiki/Matthews_correlation_coefficient)  
  - A robust metric for imbalanced multiclass classification.

- **Macro-F1 Score**  
  - [scikit-learn doc](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)

- **ROC & PR Curves**  
  - Davis, J., & Goadrich, M. (2006). *The Relationship Between Precision-Recall and ROC Curves*. [ICML](https://dl.acm.org/doi/10.1145/1143844.1143874)

### Class Imbalance Handling

- **SMOTE (Synthetic Minority Over-sampling Technique)**  
  - Chawla, N. V., et al. (2002). *SMOTE: Synthetic Minority Over-sampling Technique*. [DOI](https://doi.org/10.1613/jair.953)
  - [imbalanced-learn documentation](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)
  - Oversamples minority classes by creating synthetic examples using nearest neighbors in feature space.

- **Dynamic SMOTE Neighbors**  
  - Implemented with safe logic: `k_neighbors` chosen based on minimum class counts in split data.

### Software Stack

- **scikit-learn**  
  - [https://scikit-learn.org/](https://scikit-learn.org/)
  - Core machine learning library used for modeling, tuning, and metrics.

- **imbalanced-learn**  
  - [https://imbalanced-learn.org](https://imbalanced-learn.org/)
  - SMOTE and sampling strategies for dealing with class imbalance.

- **Python 3.10+**  
  - Project tested on Python ≥3.10

### Course & Material

- **AML-BASIC Course (2025)** — University of Bologna  
  - Official repo: [Google Drive](https://drive.google.com/drive/folders/1ZrQpF_F9E45yQTO9mG8Izr3LaECVH0aH)
---

## Author

Martina Castellucci  
AML-BASIC 2025 – University of Bologna  
martina.castellucci@studio.unibo.it

---

## License

See `LICENSE.md` for details.





