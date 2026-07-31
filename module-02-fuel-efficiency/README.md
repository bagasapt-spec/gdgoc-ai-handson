# Vehicle Fuel Efficiency Analysis and Modelling

This module contains a comprehensive exploratory data analysis (EDA), statistical visualization, and mathematical modelling framework applied to the vehicle miles-per-gallon (MPG) dataset. Developed as part of the **Google Developer Group (GDG) On Campus ITB - Artificial Intelligence Module 2 task**.

## Technical Capabilities and Analytical Findings

* **Data Profiling and Cleaning**: Implemented data cleaning methodologies to handle missing values systematically and constructed an execution profile for descriptive statistics utilizing vectorized NumPy operations.
* **Statistical Insights**: Evaluated feature relationships using Pearson correlation matrices, identifying vehicle weight as the primary negative predictor (-0.83) of fuel efficiency.
* **Mathematical Frameworks from Scratch**:
  * **Linear Regression**: Developed a single-input predictive model utilizing the analytical Normal Equation formulation, achieving mathematical parity with standard optimization libraries.
  * **Dimensionality Reduction**: Constructed a 2D Principal Component Analysis (PCA) framework via covariance matrix estimation and eigenvalue decomposition to project high-dimensional features into a lower-dimensional subspace.
* **Object-Oriented Architecture**: Designed a reusable `DatasetProfiler` class to automate numeric dataset exploration, data distribution plotting, and correlation matrix rendering within a single execution cell.

## Technical Stack

* **Programming Language**: Python
* **Core Libraries**: NumPy, Pandas, Matplotlib, Seaborn

## Module Structure

* `Hands_On_AI_Module_2.ipynb`: The primary Jupyter Notebook containing the complete source code, visualization outputs, and comprehensive markdown interpretations.
* `README.md`: Module documentation and executive summary.