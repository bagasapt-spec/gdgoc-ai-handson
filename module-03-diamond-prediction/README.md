# Diamond Price Prediction Pipeline

This module implements an end-to-end machine learning regression pipeline built with **Scikit-Learn** to predict diamond prices using the Seaborn `diamonds` dataset. The workflow spans exploratory data analysis (EDA), anomaly handling, baseline Multiple Linear Regression, polynomial degree diagnostics, categorical encoding, and K-Fold cross-validation. Developed as part of the **Google Developer Group On Campus (GDGOC) ITB - Artificial Intelligence Module 3 task**.

## Technical Capabilities and Analytical Findings

* **Data Quality & Anomaly Pruning**: Identified and removed non-physical spatial measurements ($x, y, z \le 0$) and extreme measurement outliers ($y, z > 30\text{ mm}$), dropping 23 corrupted records to prevent spatial distortion.
* **Multicollinearity Inspection**: Uncovered severe linear correlation ($r > 0.95$) among physical dimensions ($x, y, z$) and `carat`. Analyzed how collinearity inflates coefficient variance, leading to counter-intuitive negative weights in standard MLR.
* **Polynomial Regression & Overfitting Diagnostics**: Evaluated polynomial degrees 1, 2, 3, and 5 over `carat`. Demonstrated how degree 5 introduces extreme boundary divergence ($> 3.5\text{ carat}$), proving that lower-order polynomials (degree 2/3) offer superior generalizability.
* **Advanced Pipeline Feature Engineering**: Constructed a unified pipeline utilizing `ColumnTransformer` (`OneHotEncoder` + `StandardScaler`), which slashed Test RMSE by **25.00%** (from **$1,453.64** down to **$1,090.26** with $R^2 = 0.9246$).
* **Cross-Validation & Convergence Analysis**: Implemented 5-Fold Cross-Validation with explicit shuffling (`KFold(shuffle=True)`), yielding a stable mean $R^2$ score of **0.8610 ± 0.0089** and confirming convergence via learning curves.

## Business Insights for Executive Decision-Making

1. **Non-Linear Carat Premium**: Diamond valuation accelerates non-linearly with weight. A 2-carat stone commands exponentially higher market value than two 1-carat stones combined.
2. **Dimensional Proportion Scaling**: Physical proportions ($x, y$) provide essential safeguards against overvaluing heavy stones with sub-optimal cut depth or table metrics.

## Technical Stack

* **Programming Language**: Python
* **Core Libraries**: Scikit-Learn, Pandas, NumPy, Matplotlib, Seaborn

## Module Structure

* `diamond_prediction.ipynb`: The main execution notebook with complete pipeline implementation, residual plots, and learning curves.
* `test kedua.pdf`: High-resolution exported report of the executed notebook.
* `README.md`: Module documentation and executive summary.