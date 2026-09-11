# Mall Customer Segmentation & Advanced Techniques Pipeline

This module implements an end-to-end unsupervised machine learning and customer segmentation pipeline built with **Scikit-Learn** using the `Mall Customers` dataset. The workflow encompasses exploratory data analysis (EDA), geometric feature standardization, high-dimensional K-Means clustering versus PCA-reduced clustering, heuristic validation (Elbow Method and Silhouette analysis), customer persona profiling, and advanced analytical tasks including Ridge versus Lasso regularization dynamics, Agglomerative Hierarchical Clustering (dendrogram verification), and centroid initialization stability. Developed as part of the **Google Developer Group On Campus (GDGOC) ITB - Artificial Intelligence Module 5 task**.

## Technical Capabilities and Analytical Findings

* **Data Preprocessing & Distance Metric Integrity**: Dropped `CustomerID` due to its lack of analytical value. One-hot encoded `Gender` with `drop_first=True` and `dtype=int` to eliminate multicollinearity. Standardized numeric features (`Age`, `Annual Income (k$)`, and `Spending Score (1-100)`) using `StandardScaler` to prevent high-magnitude features from dominating Euclidean distance metrics in K-Means and PCA.
* **Curse of Dimensionality & PCA Projection**: Applied Principal Component Analysis (PCA) to the 4-dimensional preprocessed matrix. Identified the minimum component threshold explaining at least 90% of total variance, successfully removing linear correlation and noise while projecting the data into a more compact space for cleaner boundary separation.
* **Cluster Diagnostics & Heuristic Evaluation**:
  * **Elbow Method (WCSS)**: Assessed Within-Cluster Sum of Squares across $K \in [1, 10]$, noting a gradual decrease typical of continuous behavioral data.
  * **Silhouette Analysis**: Evaluated cluster cohesion and inter-cluster separation across $K \in [2, 10]$. Identified $K=5$ as the optimal configuration, achieving superior geometric separation and commercial interpretability compared to higher-dimensional raw representations.
* **Regularization Dynamics (Ridge vs. Lasso)**: Standardized predictors (`Age` and `Annual Income`) to predict `Spending Score` across regularization strengths $\alpha \in [10^{-2}, 10^{3}]$. Demonstrated that Ridge (L2) smoothly shrinks coefficients asymptotically toward zero, whereas Lasso (L1) drives irrelevant feature coefficients to absolute zero, serving as an embedded feature selection mechanism.
* **Topological Validation via Hierarchical Clustering**: Ran Agglomerative Clustering using Ward linkage on the standardized feature matrix. The resulting dendrogram revealed a natural cut at Ward distance $\approx 7.5$, yielding 5 distinct branches whose centroid profiles closely matched the K-Means clusters, confirming true data topology rather than algorithmic artifacts.
* **Initialization Sensitivity & Optimization Stability**: Benchmarked 10 random seeds comparing single-run K-Means (`n_init=1`) against multi-start K-Means (`n_init=10`). Proved that `n_init=1` produces significant WCSS variance due to entrapment in sub-optimal local minima, whereas `n_init=10` yields near-zero variance by selecting the run with the lowest global inertia.

## Commercial & Customer Persona Insights

1. **Young High-Spenders (Target Segment)**: Characterized by an average age under 30, moderate-to-high income, and high spending scores (>75). Responsive to fashion trends, digital lifestyle activations, and premium product releases.
2. **Careful High-Earners (Value-Driven Cohort)**: Mature segment (mean age 40+) with high annual income (>80k USD) but low spending scores (<30). Pragmatic consumers who prioritize durability, functional utility, and post-purchase service over discount campaigns.
3. **Mainstream Middle-Class (Core Volume)**: Balanced income and spending profiles in the mid-tier range. Represents the foundation for predictable daily foot traffic and essential retail goods.
4. **Budget Spenders (Value Chasers)**: Younger demographic with limited income but elevated spending frequency. Enthusiastic adopters of promotional sales, bundled discounts, and flexible payment arrangements (e.g., BNPL and 0% installment plans).
5. **Frugal Traditionalists (Infrequent Shoppers)**: Older age group exhibiting low income and very conservative spending behavior. Visits are primarily driven by specific primary necessities.

## Technical Stack

* **Programming Language**: Python
* **Core Libraries**: Scikit-Learn, Pandas, NumPy, Matplotlib, Seaborn, SciPy

## Module Structure

* `customer_segmentation.ipynb`: The primary execution notebook containing data preprocessing, PCA projection, dual-space K-Means evaluations, visual comparisons, persona profiling, and bonus experiments.
* `README.md`: Module documentation, technical summary, and business insights.