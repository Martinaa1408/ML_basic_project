# ML_basic_project – Yeast Protein Localization with Machine Learning

This project implements a complete supervised learning pipeline for predicting the **subcellular localization of yeast proteins** using physicochemical descriptors. Built for the AML-BASIC course (Applied Machine Learning, University of Bologna, 2025), it emphasizes **metric fairness**, **robust evaluation**, and **reproducibility** across all steps.

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
- **Focus**: Interpretability, generalization, class imbalance handling  
- **Tools**: `scikit-learn`, `imbalanced-learn`, `SMOTE`, `GridSearchCV`, ROC & PR analysis  

---

## Dataset

- **Source**: [UCI Yeast Dataset](https://archive.ics.uci.edu/ml/datasets/Yeast)
- **Samples**: 1,484 protein instances  
- **Classes**: 10 (e.g., CYT, MIT, NUC, POX, VAC, ME1–ME3, ERL, EXC)  
- **Features**: 8 numeric descriptors (e.g., hydrophobicity, signal peptide scores)

> Two features (`pox`, `erl`) were removed due to near-zero variance (>99.9% constant).

---

## Pipeline Summary

| Step              | Description |
|------------------|-------------|
| **EDA**           | Class distribution, feature correlation, outlier check |
| **Preprocessing** | Dropped low-variance features, scaling via `StandardScaler`, train/test split, SMOTE |
| **Tuning**        | Cross-validated hyperparameter tuning (`C` for SVM, `k` for k-NN) |
| **Modeling**      | Logistic Regression, Random Forest, SVM, k-NN |
| **Evaluation**    | Accuracy, Macro-F1, MCC, Confusion Matrix, ROC-AUC, PR-AUC |
| **Export**        | Artifacts, final model, processed data, plots, summaries |

---

## Results Overview

| Model               | Accuracy | Macro-F1 |   MCC   |
|--------------------|----------|----------|---------|
| Logistic Regression| 0.8923   | 0.7870   | 0.8642  |
| **Random Forest**   | **0.9832** | **0.9412** | **0.9785** |
| SVM (C=10)          | 0.9832   | 0.9301   | 0.9783  |
| k-NN (k=5)          | 0.8182   | 0.6302   | 0.7712  |

- **Random Forest** achieved the best overall test performance.  
- **SVM** with `C=10` was competitive and more stable across folds.  
- **k-NN** performed best in cross-validation with `k=5`, but underperformed on the test set.

---

## Model Evaluation

- Metrics: **Macro-F1**, **Matthews Correlation Coefficient (MCC)**, **ROC-AUC**, **Average Precision (AP)**
- Evaluation included:
  - **Confusion Matrix** (highlighting CYT↔MIT, POX↔NUC confusions)
  - **ROC Curve** with class labels and AUC per class (One-vs-Rest)
  - **Precision-Recall Curve** showing AP for each class
- SMOTE applied with **safe `k_neighbors`** derived dynamically from minority class sizes

---

## Project Structure

bash
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

---

## Reproducibility

This project ensures full reproducibility through:

- Saved datasets and splits in `.csv`
- Model dumps (`.pkl`) for all classifiers and GridSearchCV objects
- Plot exports (ROC, PR, Confusion Matrix)
- Modularized preprocessing scripts
- `requirements.txt` with exact library versions

---

## Author

Martina Castellucci
AML-BASIC 2025, University of Bologna  
martina.castellucci@studio.unibo.it

---

## Acknowledgments

Based on foundational work in:

**Horton, P., & Nakai, K. (1996)**  
*A Probabilistic Classification System for Predicting the Cellular Localization Sites of Proteins*  
Proceedings of the Fourth International Conference on Intelligent Systems for Molecular Biology (ISMB), pp. 109–115.  
[PDF (AAAI)](https://www.aaai.org/Papers/ISMB/1996/ISMB96-012.pdf)  
[PubMed](https://pubmed.ncbi.nlm.nih.gov/8877510)

**Nakai, K., & Kanehisa, M. (1991)**  
*Expert System for Predicting Protein Localization Sites in Gram-Negative Bacteria*  
Proteins: Structure, Function, and Genetics, 11(2), 95–110.  
[DOI](https://doi.org/10.1002/prot.340110203)  
[PubMed](https://pubmed.ncbi.nlm.nih.gov/1946347)

**Nakai, K., & Kanehisa, M. (1992)**  
*A Knowledge Base for Predicting Protein Localization Sites in Eukaryotic Cells*  
Genomics, 14(4), 897–911.  
[DOI](https://doi.org/10.1016/S0888-7543(05)80111-9)  
[PubMed](https://pubmed.ncbi.nlm.nih.gov/1478671)

Special thanks to instructors and contributors of AML-BASIC (2025 Edition).

---

## License

Distributed under **CC-BY 4.0** license.  
You are free to reuse, modify, and redistribute with attribution.

---
