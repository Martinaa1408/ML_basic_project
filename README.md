# ML_basic_project – Yeast Protein Localization with Machine Learning

[![Run full ML pipeline](https://img.shields.io/badge/GitHub-Run%20full%20ML%20pipeline-grey?logo=github)](https://github.com/) 
![Pipeline](https://img.shields.io/badge/pipeline-failing-red)
[![License](https://img.shields.io/badge/License-CC--BY--NC--SA--4.0-orange.svg)](LICENSE.md)
[![Course Repository](https://img.shields.io/badge/AML--BASIC%20Repo-View-blue)](https://drive.google.com/drive/folders/1ZrQpF_F9E45yQTO9mG8Izr3LaECVH0aH)

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

## ⚙️ Pipeline Summary

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

- ✅ **Datasets**
  - Raw dataset available in `/data/raw/`
  - Preprocessed features saved as `.csv` and `.pkl`
  - Stratified train/val/test splits stored with fixed `random_state`

- ✅ **Models**
  - All trained models (`LogisticRegression`, `RandomForest`, `SVM`, `k-NN`) saved as `.pkl`
  - Best `GridSearchCV` objects persisted for analysis
  - Parameter grids and cross-validation scores are logged

- ✅ **Metrics & Plots**
  - All performance metrics saved: `accuracy`, `macro-F1`, `MCC`, `ROC-AUC`, `PR-AUC`
  - Confusion matrices saved as both `.png` and `.pdf`
  - ROC and PR curves generated One-vs-Rest for multiclass setting

- ✅ **Pipeline**
  - Modular preprocessing scripts:
    - Feature filtering
    - Standardization
    - SMOTE augmentation with safe `k_neighbors`
  - Deterministic random seeds across `numpy`, `sklearn`, and `imblearn`

- ✅ **Environment**
  - `requirements.txt` provided with pinned versions
  - Compatible with Python 3.10+
  - Ready to be run on local or Colab environments

---

## References

- Horton, P., & Nakai, K. (1996). *Subcellular localization of proteins*. [PubMed](https://pubmed.ncbi.nlm.nih.gov/8877510)  
- Nakai, K., & Kanehisa, M. (1992). *Prediction systems for eukaryotic proteins*. [DOI](https://doi.org/10.1016/S0888-7543(05)80111-9)  
- **AML-BASIC 2025 Repository**: [Google Drive Course Folder](https://drive.google.com/drive/folders/1ZrQpF_F9E45yQTO9mG8Izr3LaECVH0aH)  
- **Lecture Notes**: *SBOBINE MACHINE LEARNING* – internal course material  

---

## Author

Martina Castellucci  
AML-BASIC 2025 – University of Bologna  
martina.castellucci@studio.unibo.it

---

## License

See `LICENSE.md` for details.





