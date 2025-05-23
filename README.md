# Yeast Protein Localization — Applied Machine Learning Project

This project explores the use of supervised machine learning techniques to predict the **subcellular localization** of yeast proteins using physicochemical features, addressing challenges such as **multiclass imbalance**, **limited features**, and **fair model evaluation**.  

Built as part of the AML-BASIC course at the University of Bologna.

---
## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [ML Pipeline](#ml-pipeline)
- [Results Summary](#results-summary)
- [Evaluation Details](#evaluation-details)
- [Repository Structure](#repository-structure)
- [Concepts from AML-BASIC](#concepts-from-aml-basic)
- [Trained Models](#trained-models)
- [Author](#author)
- [Acknowledgments](#acknowledgments)
---

## Project Overview

- **Goal**: Predict one of 10 protein localization classes from 8 numerical features
- **Type**: Supervised multiclass classification
- **Challenge**: Strong class imbalance, overlapping feature distributions, limited annotation
- **Focus**: Evaluation with macro-F1, MCC, ROC/PR curves — not accuracy alone

---

## Dataset

- **Source**: [UCI Yeast Dataset](https://archive.ics.uci.edu/ml/datasets/Yeast)
- **Samples**: 1,484 proteins  
- **Features**: 8 physicochemical descriptors (e.g., signal score, hydrophobicity)  
- **Classes**: 10 subcellular localizations (e.g., CYT, NUC, MIT, POX, ERL)

---

## ML Pipeline

| Phase             | Description |
|------------------|-------------|
| **EDA**          | Distribution, correlation, outlier analysis |
| **Preprocessing**| Standard scaling, label encoding, SMOTE (with dynamic `k_neighbors`) |
| **Modeling**     | Logistic Regression, Random Forest, SVM, k-NN |
| **Evaluation**   | Accuracy, Macro-F1, MCC, Confusion Matrix, ROC, PR Curves |
| **Error Analysis**| Biological interpretation (e.g., MIT↔CYT), limitations of small classes |

---

## Results Summary

| Model              | Accuracy | Macro F1 | Weighted F1 |
|-------------------|----------|----------|-------------|
| Baseline RF       | 0.61     | 0.46     | 0.61        |
| RF + class_weight | 0.63     | 0.49     | 0.61        |
| RF + SMOTE        | 0.65     | 0.55     | 0.64        |
| GridSearchCV RF   | **0.67** | **0.58** | **0.66**    |

ROC AUC for frequent classes ≈ 0.80–0.90  
`ERL` AUC = 1.00 (overfitting due to single test sample)

---

## Evaluation Details

- **Metrics**: macro-F1, MCC, AP (Precision-Recall), AUC (ROC)
- **Tools**: One-vs-Rest strategy for multi-class ROC/PR curves
- **Visualizations**: grouped ROC/PR plots (3-class subplots for readability)
- **Error Analysis**: misclassifications due to feature overlap (e.g., MIT vs CYT)

---

## Repository Structure

ML_basic_project/

`data/` --> Raw and processed datasets (e.g., yeast.csv or yeast.pkl)
`models/` --> Serialized models trained during the notebook (.pkl)
`notebooks/` --> Main Jupyter Notebook containing the full ML pipeline
`results/` --> Evaluation outputs (confusion matrices, ROC/PR curves, plots)
`scripts/` --> Optional Python scripts for preprocessing or utilities
`report/` --> Final written report (.pdf or .tex) for presentation/submission
`requirements.txt` --> List of required Python packages and versions
`README.md` -->  This documentation file

---

## Concepts from AML-BASIC

This project applies theoretical concepts covered in the course, including:

- **Class imbalance handling** → SMOTE, `class_weight`, macro metrics
- **Model evaluation** → bias-variance trade-off, ROC, PR curves
- **GridSearchCV** → for controlled hyperparameter optimization
- **Error analysis** → to interpret model weaknesses in domain context

_All metrics and methods are explained following the lecture notes._

---

## Trained Models

All models are saved in `/models` and ready to be reused:

- `model_baseline.pkl`
- `model_balanced.pkl`
- `model_smote.pkl`
- `model_gridsearch.pkl`

---

## Author

**Martina Castellucci**  
MSc Bioinformatics (1st year) – University of Bologna  
Course: Applied Machine Learning BASIC – Prof. Bonacorsi, Clissa (2025)

---

## Acknowledgments

- [UCI ML Repository – Yeast Dataset](https://archive.ics.uci.edu/ml/datasets/Yeast)
- [AML course slides and material(Google Drive)](https://drive.google.com/drive/folders/1ZrQpF_F9E45yQTO9mG8Izr3LaECVH0aH)

