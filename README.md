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

