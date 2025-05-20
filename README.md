# Yeast Protein Localization — Machine Learning Project

Predicting the subcellular localization of yeast proteins using supervised machine learning.  
This project builds and evaluates classifiers on the **Yeast dataset** from the UCI Machine Learning Repository, focusing on handling **class imbalance**, assessing **model performance**, and supporting **biological feature interpretation**.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Pipeline](#pipeline)
- [Repository Structure](#repository-structure)
- [Models](#models)
- [Task](#task)
- [Optional Extensions](#optional-extensions)
- [Author](#author)
- [Acknowledgments](#acknowledgments)

# 🧬 Yeast Protein Localization — Machine Learning Project

Predicting the subcellular localization of yeast proteins using supervised machine learning.  
This project builds and evaluates classifiers on the Yeast dataset from the UCI Machine Learning Repository, focusing on handling class imbalance, evaluating models beyond accuracy, and analyzing feature contributions.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Pipeline](#pipeline)
- [Repository Structure](#repository-structure)
- [Models](#models)
- [Task](#task)
- [Optional Extensions](#optional-extensions)
- [Author](#author)
- [Acknowledgments](#acknowledgments)

---

## Overview

- **Goal**: Predict the localization site of a protein (10 possible classes) based on 8 numeric biological features  
- **Domain**: Bioinformatics + Machine Learning  
- **Type of task**: Supervised, multi-class classification  
- **Challenge**: Highly imbalanced classes, overlapping features  
- **Key focus**: Robust evaluation (F1-score, ROC/PR curves), model optimization, interpretability

---

## Dataset

- **Source**: [UCI Yeast Dataset](https://archive.ics.uci.edu/ml/datasets/Yeast)  
- **Samples**: 1,484 yeast proteins  
- **Features**: 8 numerical descriptors (e.g. signal sequence score, hydrophobicity)  
- **Target**: Subcellular localization (10 classes: CYT, NUC, MIT, POX, EXC, VAC, ERL, etc.)

---

## Pipeline

1. **Exploratory Data Analysis (EDA)**  
   - Class distribution, feature correlation, imbalance analysis  
2. **Preprocessing**  
   - Label encoding, feature scaling (StandardScaler), stratified train/test split  
   - SMOTE oversampling on underrepresented classes  
3. **Model Training & Optimization**  
   - Random Forest (baseline)  
   - Random Forest with `class_weight="balanced"`  
   - SMOTE-based training  
   - GridSearchCV tuning (`n_estimators`, `max_depth`)  
4. **Evaluation**  
   - Accuracy, macro & weighted F1-scores  
   - Per-class precision & recall  
   - Confusion matrix heatmaps  
   - ROC and Precision-Recall curves (1-vs-rest)  
   - Feature importance analysis (bar plot)

---

## Repository Structure

Yeast_ML_Classification/  
│  
├── data/               # Raw and preprocessed datasets  
├── notebooks/          # Jupyter notebook(s)  
│   └── yeast_project_Notebook.ipynb  
├── models/             # Saved model files (.joblib, .pkl)  
├── results/            # Plots, confusion matrices, evaluation curves  
├── scripts/            # Reusable helper functions (optional)  
├── README.md           # Project documentation  
└── requirements.txt    # Python dependencies

---

## Models

The following models are trained and compared:

- **Baseline Random Forest**  
- **Random Forest with class weights** (`class_weight="balanced"`)  
- **Random Forest on SMOTE-balanced data**  
- **GridSearchCV-optimized Random Forest**

### Evaluation metrics:

- **Accuracy**  
- **Macro / weighted F1-score**  
- **Confusion matrix (multiclass)**  
- **ROC AUC and Precision-Recall curves** (top 3 classes)

---

## Task

- **Input**: 8 numerical features per protein  
- **Output**: One of 10 subcellular localization labels  
- **Evaluation strategy**:  
  - Train/test split (80/20, stratified)  
  - Macro-averaged metrics for class imbalance  
  - Model selection based on GridSearchCV scoring (`f1_macro`)  
  - Feature importance and interpretability

---

## Optional Extensions

To enhance analysis and model understanding:

- **Post-hoc error analysis**:  
  Identify confused class pairs (e.g., MIT vs CYT) using the confusion matrix  
  > Example: "Are mitochondrial proteins misclassified due to overlapping signal scores?"

- **Feature importance visualization**:  
  Extract and rank features using `feature_importances_` from Random Forest  

- **Dimensionality reduction** (optional):  
  Use PCA or t-SNE to visualize class clusters in 2D

---

## Author

Martina Castellucci  
MSc in Bioinformatics (1st year) — University of Bologna  
Course: *Applied Machine Learning Basic* (Prof. Daniele Bonacorsi and Luca Clissa, 2025)

---

## Acknowledgments

This project was developed as part of the *Applied Machine Learning Basic* course at the University of Bologna.  
It applies techniques learned during the course and integrates both theoretical and practical elements.

- **Dataset**: [UCI Machine Learning Repository – Yeast Dataset](https://archive.ics.uci.edu/ml/datasets/Yeast)  
- **Course Material**: *MACHINE LEARNING.pdf*  
- **Notebook & outputs**: [Google Drive Project Folder](https://drive.google.com/drive/folders/1ZrQpF_F9E45yQTO9mG8Izr3LaECVH0aH)  

Special thanks to **Prof. Daniele Bonacorsi** and **Luca Clissa** for their guidance and teaching.
