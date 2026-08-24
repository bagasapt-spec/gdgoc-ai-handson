# Titanic Survival Classification Pipeline

This module implements an end-to-end machine learning classification pipeline built with **Scikit-Learn** to predict passenger survival using the Seaborn `titanic` dataset. The workflow spans exploratory data analysis (EDA), missing value imputation, baseline Logistic Regression with log-odds interpretation, model complexity diagnostics (KNN bias-variance tradeoff & Decision Tree depth regularization), threshold tuning, ColumnTransformer pipelines, and Stratified K-Fold cross-validation. Developed as part of the **Google Developer Group On Campus (GDGOC) ITB - Artificial Intelligence Module 4 task**.

## Technical Capabilities and Analytical Findings

* **Data Quality & Missing Value Imputation**: Dropped the `deck` feature (>77% missing) and 2 missing `embarked` rows (<0.25%). Imputed missing `age` values using grouped medians by `pclass` and `sex` to preserve demographic integrity while avoiding outlier distortion. Removed redundant columns (`alive`, `class`, `embark_town`, `who`, `adult_male`).
* **Linear Classification & Log-Odds Interpretability**: Built a baseline Logistic Regression pipeline with `StandardScaler`. Derived log-odds and odds ratios, confirming that being female strongly increased survival odds ($\text{Odds Ratio} \approx 3.53$), whereas higher numerical `pclass` (3rd class) substantially lowered survival odds ($\text{Odds Ratio} \approx 0.43$).
* **Model Complexity & Overfitting Diagnostics**: 
  * **K-Nearest Neighbors**: Analyzed the bias-variance tradeoff across $K \in [1, 25]$, identifying $K=5\text{–}7$ as the sweet spot balancing local noise sensitivity (high variance at $K=1$) and global over-smoothing (high bias at large $K$).
  * **Decision Tree**: Evaluated `max_depth` constraints, observing extreme overfitting on unconstrained trees (`max_depth=None`, ~98% train accuracy vs ~78% test accuracy). Selected `max_depth=4` for optimal test generalization.
* **Cost-Sensitive Threshold Tuning & Advanced Pipelines**:
  * Utilized `precision_recall_curve` to tune operational decision thresholds, balancing F1-score optimization against a high-recall policy ($\ge 85\%$ Recall) for rescue-critical resource allocation.
  * Implemented a unified `ColumnTransformer` (`OneHotEncoder` on categorical/ordinal features + `StandardScaler` on numerical features), enabling distinct non-linear weight assignments for each passenger class.
* **Stratified Cross-Validation & Metric Stability**: Executed 5-Fold Stratified Cross-Validation (`StratifiedKFold`), demonstrating consistent ROC-AUC performance across folds and eliminating sampling bias caused by target distribution variations.

## Historical & Stakeholder Insights

1. **Evacuation Priority by Gender**: The historical "women and children first" maritime evacuation protocol is starkly visible in the data, with female passengers achieving an overall survival rate nearly four times higher than male passengers.
2. **Socioeconomic Class Disparities**: First-class passengers survived at more than double the rate of third-class passengers, directly tied to upper-deck cabin locations and prioritized access to lifeboats.
3. **Family Grouping Dynamics**: Solo travelers and passengers in large family units experienced lower survival rates compared to small family units (1–2 dependents), where mobility and mutual assistance were optimal.

## Technical Stack

* **Programming Language**: Python
* **Core Libraries**: Scikit-Learn, Pandas, NumPy, Matplotlib, Seaborn

## Module Structure

* `titanic_classification.ipynb`: The main execution notebook featuring full EDA, multi-model evaluation, ROC/PR curves, and bonus implementations.
* `README.md`: Module documentation and executive summary.