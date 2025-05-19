# Yeast Protein Localization — Machine Learning Project

Predicting the subcellular localization of yeast proteins using supervised machine learning.  
This project builds and evaluates classifiers on the **Yeast dataset** from the UCI Machine Learning Repository, focusing on handling **class imbalance**, assessing **model performance**, and supporting **biological feature interpretation**.

---

## Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Pipeline](#-pipeline)
- [Repository Structure](#-repository-structure)
- [Models](#-models)
- [Task](#-task)
- [Optional Extensions](#-optional-extensions)
- [Author](#-author)
- [Acknowledgments](#-acknowledgments)

---

## Overview

- **Goal**: Predict the localization site of a protein (10 possible classes) based on 8 numeric biological features  
- **Domain**: Bioinformatics + Machine Learning  
- **Type of task**: Supervised, multi-class classification  
- **Challenge**: Imbalanced classes, limited features, domain-specific interpretation  
- **Key focus**: Robust evaluation (F1-score, precision/recall), not just accuracy

---

## Dataset

- **Source**: [UCI Yeast Dataset](https://archive.ics.uci.edu/ml/datasets/Yeast)
- **Samples**: 1,484 yeast proteins  
- **Features**: 8 numeric descriptors (e.g. signal score, hydrophobicity, etc.)  
- **Target**: Subcellular localization (10 classes: CYT, NUC, MIT, POX, EXC, etc.)

---

## Pipeline

1. **Exploratory Data Analysis (EDA)**  
   - Class distribution, outliers, correlation  
2. **Preprocessing**  
   - Feature scaling, encoding, train/val/test split, optional SMOTE  
3. **Model Training & Tuning**  
   - Logistic Regression, Random Forest, SVM, k-NN  
   - 5/10-fold Cross-validation  
4. **Evaluation**  
   - Confusion matrix, F1-score (macro), per-class metrics  
   - Learning curves, bias-variance analysis  
5. **Optional: Post-hoc error analysis**  
   - Investigate most confused class pairs (e.g., EXC vs. CYT)

---

## Repository Structure

Yeast_ML_Classification/
│
├── data/ # Raw and preprocessed datasets
├── notebooks/ # Jupyter notebooks by stage
│ ├── 01_EDA.ipynb
│ ├── 02_Preprocessing.ipynb
│ ├── 03_Training.ipynb
│ ├── 04_Evaluation.ipynb
├── models/ # Saved model files (.joblib, .pkl)
├── results/ # Outputs: plots, confusion matrices, metrics
├── scripts/ # Reusable functions (optional)
├── README.md # Project documentation
└── requirements.txt # Dependencies

---

## Models

The following models are tested and compared:

- **Logistic Regression** (One-vs-Rest strategy)  
- **Random Forest Classifier** (for robustness and feature importance)  
- **Support Vector Machine** (linear or RBF kernel)  
- **k-Nearest Neighbors** (as baseline)

Evaluation is based on:
- F1-score (macro average)
- Per-class precision & recall
- Confusion matrix
- Optional ROC-AUC curves

---

## Task

- **Input**: 8 numerical features per protein  
- **Output**: One of 10 subcellular locations  
- **Evaluation strategy**:  
  - 5 or 10-fold cross-validation  
  - Separate test set  
  - Analysis of learning curves  
  - Interpretation of confusion matrix

---

## Optional Extensions

To enhance analysis and project quality:

- **Error Analysis**:  
  Identify and analyze pairs of classes that are frequently confused (e.g., MIT vs. CYT). Plot confusion matrix heatmaps and investigate feature distributions.  
  > Example: "Are mitochondrial proteins misclassified due to similar signal scores?"

- **Feature importance visualization**:  
  Extract and rank feature contributions from tree-based models (Random Forest).  

- **Dimensionality reduction for visualization**:  
  Apply PCA or t-SNE to project data into 2D space and visually inspect class separability.

---

## Author

Martina Castellucci  
MSc in Bioinformatics (1st year) — University of Bologna  
Course: Applied Machine Learning Basic (Prof. Daniele Bonacorsi and Luca Clissa, 2025)

---

## Acknowledgments

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Yeast)  
- Applied ML course materials (SBOBINE MACHINE LEARNING.pdf)  
- GitHub projects by [dcarbini](https://github.com/dcarbini) — reference notebook structure  
- Base notebooks and guidance from [Google Drive Project Folder](https://drive.google.com/drive/folders/1Uxq0pH5Y-y1x-af4iGqFjkxoZ8IWQhRJ)
