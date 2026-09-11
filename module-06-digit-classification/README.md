# Handwritten Digit Classification Pipeline with MLP & Keras

This module implements an end-to-end deep learning classification pipeline built with **Scikit-Learn** and **TensorFlow/Keras** using the `load_digits` dataset[cite: 2]. The workflow encompasses exploratory visual data analysis, feature standardization, baseline Multi-Layer Perceptron (MLP) training with learning curve diagnostics (overfitting detection), regularization via early stopping, hyperparameter optimization using GridSearchCV, architecture replication in Keras Sequential, misclassification structural analysis, and an architecture-regularization grid sweep[cite: 2]. Developed as part of the **Google Developer Group On Campus (GDGOC) ITB - Artificial Intelligence Module 6 task**[cite: 2].

## Technical Capabilities and Analytical Findings

* **Data Ingestion & Feature Standardization**: Processed 1,797 grayscale 8x8 pixel images (64 feature dimensions across 10 digit classes)[cite: 2]. Applied a 70/30 stratified train-test split (`stratify=y`, `random_state=42`) to preserve uniform class distributions[cite: 2]. Fitted `StandardScaler` strictly on training data to prevent data leakage across validation folds[cite: 2].
* **Baseline MLP & Overfitting Diagnostics**: Built a baseline `MLPClassifier` with architecture `(128, 64)` running for 300 epochs[cite: 2]. Extracted `loss_curve_` and `validation_scores_` to observe learning dynamics[cite: 2]. Training loss smoothly approached zero, while validation accuracy plateaued around epoch 50 to 70[cite: 1, 2]. Continued training beyond this plateau caused memorization of training sample artifacts, demonstrating empirical overfitting[cite: 1, 2].
* **Regularization & Optimizer Dynamics**:
  * **Early Stopping**: Enabling `early_stopping=True` with `n_iter_no_change=10` automatically halted training once validation accuracy ceased to improve, conserving computational budget while preserving peak generalization weights[cite: 1, 2].
  * **Adam vs SGD**: The Adam optimizer converged significantly faster due to adaptive per-parameter learning rates and momentum, whereas Stochastic Gradient Descent (SGD) exhibited slower convergence across identical epoch limits[cite: 1, 2].
* **Hyperparameter Optimization via GridSearchCV**: Evaluated combinations across `hidden_layer_sizes` (`(64, 32)`, `(100,)`), `activation` (`'relu'`, `'tanh'`), and `alpha` (`0.0001`, `0.01`) using 3-fold cross-validation[cite: 2]. Identified the optimal capacity balance that maximized test generalization score[cite: 2].
* **Keras Sequential Mirror & EarlyStopping Callback**: Dynamically mirrored the best Scikit-learn architecture inside TensorFlow/Keras using `sparse_categorical_crossentropy` and Softmax output activation[cite: 1, 2]. Implemented `tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)` to reproduce Scikit-learn regularization dynamics in production deep learning workflows[cite: 2].
* **Misclassification Topology Analysis**: Filtered non-diagonal peaks from the confusion matrix to identify the most confused digit pair[cite: 2]. Visualized 8x8 reconstructed pixel arrays, isolating structural ambiguity caused by subtle pixel intensity shifts in low-resolution handwriting[cite: 2].
* **Architecture and Alpha Sweep**: Benchmarked a 3x3 grid combining model capacity (`(32,)`, `(64, 32)`, `(128, 64)`) and L2 penalty (`0.0001`, `0.01`, `1.0`)[cite: 2]. Mapped test accuracy and train-test performance gaps into heatmaps, clearly delineating underfitting zones (excessive regularization), overfitting zones (high capacity with negligible penalty), and optimal generalization boundaries[cite: 1, 2].

## Core Analytical Insights

1. **Early Stopping as an Essential Regularizer**: Halting training based on validation loss plateaus prevents networks from over-adapting to idiosyncrasies in small training sets, yielding equivalent or superior test accuracy compared to unconstrained runs[cite: 1, 2].
2. **Structural Limitations of Flattened MLPs**: Treating 2D images as 1D flat vectors discards spatial pixel relationships[cite: 1, 2]. On 8x8 images, single-pixel variations often dictate whether an ambiguous loop resembles a digit 1, 8, or 9[cite: 1, 2].
3. **Capacity versus L2 Regularization Trade-off**: Deep architectures require proportional weight penalties (`alpha`) to constrain parameter freedom; overly strong penalties suppress representation capacity, causing underfitting regardless of depth[cite: 1, 2].

## Technical Stack

* **Programming Language**: Python
* **Core Libraries**: Scikit-Learn, TensorFlow/Keras, NumPy, Pandas, Matplotlib, Seaborn

## Module Structure

* `digit_classification.ipynb`: Complete execution notebook with data preprocessing, baseline diagnostics, hyperparameter search, confusion matrices, Keras replication, and bonus analytical sweeps.
* `README.md`: Module documentation, theoretical justification, and architecture findings.