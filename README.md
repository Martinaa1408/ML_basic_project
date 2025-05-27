# ML_basic_project – Yeast Protein Localization with Machine Learning

This project explores supervised machine learning techniques to predict the **subcellular localization** of yeast proteins using physicochemical features. It addresses the challenges of **class imbalance**, **low-dimensional input**, and **robust model evaluation**, following the principles taught in the AML-BASIC course at the University of Bologna.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [ML Pipeline](#ml-pipeline)
- [Results Summary](#results-summary)
- [Evaluation Details](#evaluation-details)
- [Repository Structure](#repository-structure)
- [Models](#models)
- [Installation](#installation)
- [Author](#author)
- [Acknowledgments](#acknowledgments)

---

## Project Overview

- **Goal**: Predict the subcellular localization of yeast proteins
- **Type**: Multiclass supervised classification
- **Challenge**: High class imbalance, overlapping feature distributions
- **Focus**: Interpretability, metric fairness, and model robustness

---

## Dataset

- **Source**: [UCI Yeast Dataset](https://archive.ics.uci.edu/ml/datasets/Yeast)
- **Samples**: 1,484 yeast proteins
- **Features**: 8 physicochemical descriptors
- **Classes**: 10 subcellular locations (e.g., CYT, NUC, MIT, POX, ERL)

---

## ML Pipeline

| Phase             | Description |
|------------------|-------------|
| **EDA**          | Class distribution, correlation, outlier analysis |
| **Preprocessing**| Encoding, scaling, SMOTE (with adaptive `k_neighbors`) |
| **Modeling**     | Logistic Regression, Random Forest, SVM, k-NN |
| **Evaluation**   | Accuracy, Macro-F1, MCC, Confusion Matrix, ROC/PR |
| **Error Analysis**| Class confusion (e.g., MIT↔CYT), biological insight |

---

## Results Summary

| Model               | Accuracy | Macro F1 | Weighted F1 |
|--------------------|----------|----------|-------------|
| Logistic Regression| 0.61     | 0.44     | 0.60        |
| Random Forest       | 0.67     | 0.58     | 0.66        |
| SVM                | 0.63     | 0.53     | 0.64        |
| k-NN               | 0.59     | 0.42     | 0.59        |

ROC AUC for frequent classes ≈ 0.80–0.90  
ERL AUC = 1.00 (likely overfitting due to 1 test sample)

---

## Evaluation Details

- **Metrics**: Macro-F1, Weighted-F1, MCC, AUC, Average Precision
- **Confusion Matrix**: Used to detect class confusion patterns
- **ROC/PR Curves**: Single unified plots displaying all one-vs-rest class comparisons together
- **SMOTE**: Applied with dynamic `k_neighbors` based on minority class size

---

## Repository Structure

`ML_basic_project/`

`data/` --> Raw and processed datasets (e.g., yeast.names, splits, .pkl)

`models/` --> Trained and serialized models (.pkl or .joblib)

`notebooks/` --> Jupyter notebook containing the full ML pipeline

`results/` --> Evaluation outputs: plots, confusion matrices, ROC/PR curves

`scripts/` --> Python scripts (e.g., preprocessing.py for SMOTE, scaling)

`report/` --> Final report (.pdf or .tex) for submission or presentation

`requirements.txt`--> List of required Python packages

`README.md` --> This documentation file


---

## Models

The following models were trained, compared, and saved:

- **Logistic Regression**
- **Random Forest** (baseline, balanced, SMOTE, GridSearchCV)
- **Support Vector Machine**
- **k-Nearest Neighbors**

All models were evaluated using macro-F1, MCC, and ROC/PR curves.  
Saved in `models/` as:

- `model_logreg.pkl`
- `model_randomforest.pkl`
- `model_svm.pkl`
- `model_knn.pkl`
- `model_gridsearch.pkl`

---

## Installation

To recreate the environment, run: `pip install -r requirements.txt`

---

## Author

**Martina Castellucci**  
MSc Bioinformatics (1st year) – University of Bologna  
Course: Applied Machine Learning BASIC – Prof. Bonacorsi, Clissa (2025)

---

## Acknowledgments

- [UCI ML Repository – Yeast Dataset](https://archive.ics.uci.edu/ml/datasets/Yeast)
- [AML course slides and material(Google Drive)](https://drive.google.com/drive/folders/1ZrQpF_F9E45yQTO9mG8Izr3LaECVH0aH)

