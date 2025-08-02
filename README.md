# Predicting Yeast Protein Localization with Machine Learning (AML-BASIC 2025)

[![ML pipeline](https://img.shields.io/badge/GitHub-Run%20full%20ML%20pipeline-grey?logo=github)](https://github.com/Martinaa1408/ML_basic_project/)
[![Notebook](https://img.shields.io/badge/Notebook-ipynb-brightgreen?logo=Jupyter)](notebooks/AML_notebook.ipynb)
[![Scripts](https://img.shields.io/badge/Scripts-Python-blue?logo=python)](scripts/)
[![License](https://img.shields.io/badge/License-CC--BY--NC--SA--4.0-orange?logo=creativecommons)](LICENSE.md)
[![Dataset](https://img.shields.io/badge/Dataset-Yeast-orange?logo=databricks)](https://archive.ics.uci.edu/ml/datasets/Yeast)
[![Course](https://img.shields.io/badge/AML--BASIC-2025-informational?logo=book)](https://drive.google.com/drive/folders/1ZrQpF_F9E45yQTO9mG8Izr3LaECVH0aH)
[![Reproducible](https://img.shields.io/badge/Reproducible-Yes-brightgreen)](requirements.txt)

<img width="517" height="352" alt="image" src="https://github.com/user-attachments/assets/a95d4826-c323-44ba-bf87-5b7e57fb0401" />

---

## Table of Contents

- [Project Overview](#project-overview)
- [Problem Framing & Pipeline Design](#problem-framing--pipeline-design)
- [Dataset](#dataset)
- [Pipeline Summary](#pipeline-summary)
- [Model Overview](#model-overview)
- [Evaluation & Metrics](#evaluation--metrics)
- [Project Structure](#project-structure)
- [Alignment with AML-BASIC 2025 Course Material](#alignment-with-aml-basic-2025-course-material)
- [References](#references)
- [Author](#author)
- [License](#license)

---

## Project Overview

This project builds a full machine learning pipeline to classify **yeast proteins** into **10 subcellular compartments** based on numeric sequence features.

It demonstrates:

- **Robust model training** on a highly imbalanced dataset with multiple target classes  
- **Fair performance assessment** using **Macro-F1**, **MCC**, and **PR-AUC**  
- **Fully reproducible workflow**, from preprocessing and SMOTE resampling to model selection and visualization

The pipeline provides a clear example of **designing and evaluating classifiers** for **biological data with skewed label distributions**.

---

## Problem Framing & Pipeline Design

This is a **multiclass classification task**, where the input is a vector of numeric protein descriptors and the output is one of 10 subcellular compartments.

The pipeline includes:
- Feature scaling and low-variance filtering
- SMOTE oversampling with dynamic `k_neighbors` applied only to the training set
- Hyperparameter tuning via `GridSearchCV`
- Evaluation with metrics robust to imbalance: **Macro-F1**, **MCC**, **PR-AUC**

---

## Dataset

- **Source**: [UCI Yeast Dataset](https://archive.ics.uci.edu/ml/datasets/Yeast)  
- **Instances**: 1,484 yeast proteins  
- **Classes**: 10 localizations (CYT, MIT, NUC, POX, VAC, ME1–ME3, ERL, EXC)  
- **Features**: 8 numeric descriptors → 6 retained after removing `pox` and `erl` (low variance)
- 
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

**Why These Models**?

- **Logistic Regression**: interpretable baseline, low complexity  
- **Random Forest**: robust, handles non-linearity, low variance  
- **SVM**: strong generalization, margin maximization  
- **k-NN**: intuitive but sensitive to scale and `k`  

---

## Evaluation & Metrics

- **Macro-F1**: class-wise balanced F1 average  
- **Matthews Correlation Coefficient (MCC)**: balanced multiclass correlation  
- **ROC-AUC** (OvR) and **PR-AUC** curves  
- **Confusion Matrices**: clear CYT↔MIT and POX↔NUC confusions  
- **SMOTE**: dynamically adjusted `k_neighbors` for minority classes  

---

## Project Structure

#### data/ — Dataset Folder

This folder contains all the data files used in the project, including the original dataset, preprocessed versions, and train-test splits.

- **`yeast.csv`**  
  Original raw dataset downloaded from the UCI Machine Learning Repository.  
  It contains 1,484 proteins, each described by 8 numerical features and a target class representing subcellular localization.

- **`yeast_dataset_processed.csv`**  
  Preprocessed version of the dataset, with all features cleaned and ready for modeling.  
  It may include standardized values, encoded labels, or filtered features based on variance or correlation.

- **`yeast_dataset_processed.pkl`**  
  Same as above, but stored as a serialized Python object using `pickle`.  
  Useful for fast loading without repeating preprocessing steps.

- **`X_train.csv`, `X_test.csv`**  
  Feature matrices for training and testing. Each row represents a protein, and each column a numeric feature.

- **`y_train.csv`, `y_test.csv`**  
  Target labels for training and testing, indicating the protein's subcellular localization class.

#### models/ — Trained Models

This folder contains the final models trained during the project, serialized using `pickle` or `joblib`.

- **`model_logreg.pkl`** — Trained Logistic Regression model (baseline).  
- **`model_randomforest.pkl`** — Trained Random Forest classifier.  
- **`model_svm.pkl`** — Trained Support Vector Machine model with optimized hyperparameters.  
- **`model_knn.pkl`** — Trained k-Nearest Neighbors model (k=5).  
- **`model_gridsearch.pkl`** — `GridSearchCV` object containing cross-validation results and best parameters.


#### scripts/ — Utility Functions

Python modules used for preprocessing, balancing, and transformations.

- **`preprocessing.py`**  
  Contains key preprocessing utilities:  
  - `scale_features(X)` – standardizes features using `StandardScaler`.  
  - `apply_safe_smote(X, y)` – applies SMOTE with adaptive `k_neighbors` to balance classes.  
  - `binarize_labels(y, class_labels)` – converts multi-class labels into binary (one-vs-rest) format for multi-label tasks.

#### results/ — Visualizations and Evaluation

Visual outputs and summary files used to evaluate model performance.

- **`roc_all_classes.png`** — ROC curves (one-vs-rest) per class using Random Forest. All classes achieve AUC = 1.00.
- **`pr_all_classes.png`** — Precision-Recall curves for each class. Most classes approach AP = 1.00.
- **`conf_matrix_rf_real.png`** — Confusion matrix for the Random Forest model, showing excellent classification accuracy.
- **`class_distribution.png`** — Histogram showing class imbalance across the 10 subcellular location classes.
- **`class_distribution_after_smote.png`** — Histogram of class frequencies in the training set after SMOTE oversampling, showing balanced classes.
- **`feature_distribution_errors.png`** — Boxplots showing outliers in selected features.
- **`feature_correlation_matrix.png`** — Heatmap of feature correlations (Pearson).
- **`model_performance_summary.png`** — Comparison of macro-F1 scores (with error bars) for different models (SVM, k-NN).
- **`comparison_table.csv`** — Table of evaluation metrics (accuracy, F1, MCC) for all trained models.
- **`summary.txt`** — Text summary of final model performance.

#### report/ — Final Report

This folder contains the official project report written in LaTeX.

- **`AML_report.pdf`**  — Compiled PDF version of the report.

#### notebooks/ — Jupyter Notebook

Notebook with step-by-step data analysis, model training, and evaluation.

- **`AML_notebook.ipynb`** — Full analysis pipeline in notebook format.  

#### requirements.txt

List of Python packages required to run the project.

You can install all dependencies using:

```bash
pip install -r requirements.txt
```
---

## Alignment with AML-BASIC 2025 Course Material

All components of this project directly reflect the structure and methods taught in AML-BASIC 2025. The dataset used (UCI Yeast) was prepared as in the course notebooks, with low-variance features removed (`pox`, `erl`) and class distribution analyzed. Preprocessing steps—standardization, stratified splitting, and label encoding—followed the Data Preparation guidelines. Class imbalance was handled using SMOTE applied only to the training set, with dynamic `k_neighbors` and selective oversampling, exactly as discussed in the resampling module. The selected models (Logistic Regression, SVM, k-NN, Random Forest) and their tuning procedures (GridSearchCV) mirror the modelling notebooks. Evaluation relied on fairness-aware metrics (Macro-F1, MCC, PR-AUC), as recommended, with a critical discussion of ROC-AUC limitations under imbalance. Finally, the project adheres to all reproducibility and modularity principles emphasized throughout the course. Every decision is consistent with the official AML-BASIC lectures, notebooks, and theoretical material.

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

This project is released under a [Creative Commons BY-NC-SA 4.0 License](LICENSE.md).
