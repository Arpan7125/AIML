# Lab 4 Report: KNN Classification & Evaluation Metrics
## Part 1: Comprehensive Study of K-Nearest Neighbours (KNN) Classification and Comparison with Regression Evaluation Metrics

**Course:** AIML Lab  
**Lab Assignment:** 04  
**Date:** July 7, 2026  

---

### Aim
To implement K-Nearest Neighbors (KNN) classification on the Breast Cancer Wisconsin dataset, optimize the hyperparameter $K$ using train-test split variation, heuristic $K$ selection, and 10-fold cross-validation. Additionally, to evaluate the final model using classification metrics (Accuracy, Precision, Recall, F1 Score, Confusion Matrix, and ROC-AUC) and perform a comprehensive comparative study with the regression evaluation metrics (MAE, MSE, RMSE, $R^2$) studied in Lab 3.

---

## 1. Task 1: Data Preparation
The Breast Cancer Wisconsin (Diagnostic) dataset contains physical measurements of cell nuclei from breast mass aspirates.
- **Samples ($n$):** 569
- **Features ($d$):** 30 numerical features (computed from digitized images)
- **Target Classes:** 
  - `0` $\rightarrow$ Malignant (212 samples, 37.26%)
  - `1` $\rightarrow$ Benign (357 samples, 62.74%)

### 1.1 Exploratory Data Analysis & Integrity
- **Missing Values:** 0 (none detected)
- **Duplicates:** 0 (none detected)
- **Class Imbalance:** Benign cases constitute 62.74% of the dataset, while malignant cases make up 37.26%. While not severely imbalanced, this distribution requires stratified splits to maintain class proportions in train and test sets.

### 1.2 Importance of Feature Scaling
KNN calculates the similarity between data points using distance metrics (typically Euclidean or Manhattan). 
- **Euclidean Distance Formula:**
  $$d(p, q) = \sqrt{\sum_{i=1}^d (p_i - q_i)^2}$$
- **The Scale Problem:** Features in this dataset have vastly different scales. For instance, `mean area` ranges from $143.5$ to $2501.0$, while `mean smoothness` ranges from $0.053$ to $0.163$. Without scaling, a difference of $10$ units in `mean area` would dwarf a difference of $0.1$ units in `mean smoothness` in the distance calculation.
- **Standardization:** We applied `StandardScaler` to normalize the data:
  $$x_{\text{scaled}} = \frac{x - \mu}{\sigma}$$
  This centers each feature around a mean of 0 ($\mu=0$) and scales it to unit variance ($\sigma^2=1$), ensuring all 30 features contribute equally to neighbor identification.

---

## 2. Task 2: Train-Test Split Analysis
To understand how dataset division affects model stability and generalization, we evaluated three split ratios (**80:20**, **70:30**, and **90:10**). The hyperparameter $K$ for each split was set using the heuristic $K = \text{round}(\sqrt{n_{\text{train}}})$, adjusted to the nearest odd integer to prevent voting ties.

### 2.1 Empirical Results across Split Ratios

| Split Ratio | Train Size | Test Size | Heuristic $K$ | Train Accuracy | Test Accuracy | Precision | Recall | F1 Score |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **80:20** | 455 | 114 | 21 | 96.48% | 95.61% | 94.67% | 98.61% | 96.60% |
| **70:30** | 398 | 171 | 21 | 96.98% | 94.74% | 92.24% | 100.00% | 95.96% |
| **90:10** | 512 | 57 | 23 | 96.29% | 98.25% | 97.30% | 100.00% | 98.63% |

### 2.2 Model Stability and Generalization Analysis
- **Model Stability (Variance):** The **90:10 split** achieved the highest test accuracy (98.25%). However, because the test set is extremely small ($n_{\text{test}} = 57$), the evaluation is highly unstable. A single misclassification in the test set changes the accuracy by $\approx 1.75\%$, making it vulnerable to random split noise.
- **Model Bias:** The **70:30 split** has the lowest test accuracy (94.74%). Reducing the training set size to 398 samples increases the model's bias, meaning the model cannot capture the feature boundaries as effectively as it can with larger training sets.
- **Optimal Balance:** The **80:20 split** represents a robust compromise. The test set ($n_{\text{test}} = 114$) is large enough to provide a stable, low-variance evaluation of generalization, and the training set ($n_{\text{train}} = 455$) is large enough for the KNN classifier to learn representative decision boundaries.

---

## 3. Task 3: KNN Model with Heuristic K Selection

### 3.1 Heuristic Selection Method
The standard rule of thumb for selecting $K$ is:
$$K_{\text{heuristic}} = \text{round}(\sqrt{n_{\text{train}}})$$
For the 80:20 split, $n_{\text{train}} = 455$.
- $\sqrt{455} = 21.33$
- Nearest odd integer $\rightarrow K = 21$ (to break ties).

### 3.2 Accuracy vs. K Value Trend
We trained KNN classifiers for $K$ values ranging from 1 to 35. The resulting accuracies on the training and test sets are plotted below:

![Accuracy vs K](images/accuracy_vs_k.png)

- **Underfitting (Large $K$):** As $K \rightarrow 35$, both train and test accuracies degrade. The neighborhood becomes too large, diluting the local structure of the data.
- **Overfitting (Small $K$):** For $K = 1$, training accuracy is 100%, but the test accuracy is lower, indicating that the model is fitting individual noise points in the training set.
- **Heuristic Range Search ($K \pm 5$):** Exploring the range $K \in [16, 26]$ around the heuristic $K=21$ reveals that the test accuracy peaks at $K = 18$ ($98.25\%$). However, since 18 is an even number, neighboring odd values like $K=17$ and $K=19$ are preferred to avoid ties.

### 3.3 Distance Metrics & Decision Boundary Mapping

#### Comparison of Distance Metrics
1. **Euclidean Distance:**
   $$d(\mathbf{p}, \mathbf{q}) = \sqrt{\sum_{i=1}^d (p_i - q_i)^2}$$
   - **Suitability:** Best when features are continuous, isotropic (equal variance), and low-to-medium dimensional.
   - **Limitations:** Highly sensitive to outlier features and suffers from the "curse of dimensionality" (distances become uniform in high-dimensional space).
2. **Manhattan Distance:**
   $$d(\mathbf{p}, \mathbf{q}) = \sum_{i=1}^d |p_i - q_i|$$
   - **Suitability:** Best for grid-based data structures, categorical features, or high-dimensional spaces where it helps maintain distance discriminability.
   - **Benefits:** Less sensitive to individual large outliers compared to Euclidean distance because it does not square the differences.

#### Decision Boundary Visualization
Since we cannot plot a 30D decision space, we projected the features onto a 2D space using Principal Component Analysis (PCA). We then trained a KNN model on the 2D PCA projection and mapped the decision boundaries for $K = 1, 5, 10, 20$.

![Decision Boundaries](images/decision_boundaries.png)

- **$K = 1$:** Jaggard, fragmented decision boundaries. The classifier overfits by creating small "islands" around individual noise data points.
- **$K = 5$ & $K = 10$:** The boundary becomes significantly smoother. The classifier filters out local noise and models the general separation boundary of the two classes.
- **$K = 20$:** The boundary is highly smoothed and simplified. While it has low variance, it risks underfitting by ignoring minor, potentially valid local patterns.

---

## 4. Task 4: Cross-Validation
To eliminate the partition bias of a single train-test split, we performed **10-Fold Cross-Validation** over all $K$ values.

![Cross-Validation Accuracy vs K](images/cv_accuracy_vs_k.png)

- **Cross-Validation Optimum:** 10-Fold CV identifies **$K = 11$** as the hyperparameter yielding the highest mean validation accuracy of **$97.01\% \pm 1.96\%$**.
- **Comparison:** The single train-test split accuracy peaked at $K=18$ (accuracy of $98.25\%$) due to the specific data points in the test set. Cross-validation averages performance over 10 distinct test folds, identifying $K=11$ as the most robust, generalizable hyperparameter.
- **Final Selection:** We select **$K = 11$** as our optimal hyperparameter, as it is an odd integer, is close to the heuristic range, and exhibits the highest mean CV accuracy.

---

## 5. Task 5: Classification Evaluation
We trained the final KNN model ($K = 11$) on the 80:20 split and evaluated its performance.

### 5.1 Final Model Performance Metrics
- **Accuracy:** 97.37% (111 out of 114 samples correctly classified)
- **Precision (Benign):** 96.00% (72 out of 75 benign predictions were correct)
- **Recall (Benign):** 100.00% (all 72 actual benign cases were correctly identified)
- **F1 Score (Benign):** 97.96%
- **ROC-AUC Score:** 0.9922 (indicating excellent class separation capability)

#### Classification Report Detail:
```text
              precision    recall  f1-score   support

   malignant       1.00      0.93      0.96        42
      benign       0.96      1.00      0.98        72

    accuracy                           0.97       114
   macro avg       0.98      0.96      0.97       114
weighted avg       0.97      0.97      0.97       114
```

### 5.2 Confusion Matrix

![Confusion Matrix](images/confusion_matrix.png)

- **True Negatives (Malignant correctly identified):** 39
- **True Positives (Benign correctly identified):** 72
- **False Positives (Malignant misclassified as Benign):** 3
- **False Negatives (Benign misclassified as Malignant):** 0

> [!IMPORTANT]
> In terms of malignant cancer detection (taking "Malignant" as the target class):
> - **Recall (Sensitivity) for cancer:** $\frac{\text{True Malignant}}{\text{Actual Malignant}} = \frac{39}{39 + 3} = 92.86\%$.
> - **False Negatives (Missed cancer cases):** 3 cases. This is the critical number in a medical diagnosis system since missed malignant cases represent a severe threat to patient survival.

### 5.3 ROC Curve and AUC Score

![ROC Curve](images/roc_curve.png)

The Area Under the Curve (AUC) of **0.9922** demonstrates that the model maintains an exceptionally high True Positive Rate across almost all False Positive Rate thresholds, signifying a highly robust classifier.

---

## 6. Task 6: Comparative Study with Regression (Lab 3 Integration)

In Lab 3 (Linear Regression), models were evaluated using Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and the $R^2$ Score. In this lab, we evaluated a KNN Classifier.

### 6.1 Conceptual Framework Comparison
- **Error-Based Evaluation (Regression):** Focuses on the **magnitude of deviation** of a continuous prediction from the ground truth. The error has a scale (e.g., predicted salary vs. actual salary).
- **Decision-Based Evaluation (Classification):** Focuses on the **correctness of a categorical choice**. It cares about which side of a decision boundary a sample falls on, rather than the scale of the difference.

### 6.2 Metric vs. Metric Comparison

#### R² Score vs. Accuracy
- **$R^2$ Score (Coefficient of Determination):** Measures the proportion of variance in the continuous dependent variable explained by the features, relative to a baseline model that predicts the mean. It ranges from $-\infty$ to $1$.
- **Accuracy:** Measures the simple fraction of correct categorical decisions. It is bounded between $0$ and $1$.
- **Comparison:** While both act as overall performance baselines, $R^2$ is sensitive to the scale of prediction errors, whereas Accuracy is sensitive only to the number of correct assignments. Both can be misleading in the presence of outliers (for $R^2$) or class imbalance (for Accuracy).

#### RMSE vs. F1 Score
- **RMSE:** Computes the square root of average squared errors, penalizing larger errors quadratically. It represents error in the target's physical unit.
- **F1 Score:** The harmonic mean of Precision and Recall:
  $$\text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
- **Comparison:** RMSE penalizes continuous prediction variance (large errors hurt more). F1 Score penalizes misclassifications (False Positives and False Negatives) symmetrically, ignoring model confidence or error magnitude, balancing class detection rate and precision.

#### MAE vs. Confusion Matrix
- **MAE:** The average of absolute errors, providing a linear, intuitive measure of prediction deviation.
- **Confusion Matrix:** A tabular layout showing correct and incorrect classifications across all combinations of true and predicted classes.
- **Comparison:** MAE condenses continuous error magnitude to a single linear value. The Confusion Matrix does not condense information; it exposes the asymmetry of errors (FP vs. FN), which is crucial for decision-making (e.g. medical vs. financial risk).

---

## 7. Detailed Inference

### 7.1 Regression Metrics and Prediction Error Magnitude
Regression metrics assess the *distance* between predicted and true values on a continuous scale (e.g., predicting tumor volume). An error of $1 \text{ cm}^3$ is better than an error of $10 \text{ cm}^3$. Regression metrics evaluate continuous numeric precision.

### 7.2 Classification Metrics and Decision Correctness
Classification metrics assess whether a sample was assigned the correct categorical label. If a patient is diagnosed as "Malignant," the classification framework treats it as a single binary decision (Correct/Incorrect). The model's internal probability (e.g., predicting Malignant with 51% confidence vs. 99% confidence) is secondary to the final categorical decision.

### 7.3 Insufficiency of Accuracy in Medical Diagnosis
Accuracy treats all correct predictions equally and fails under class imbalance. In cancer screening, if only 1% of patients have cancer, a model that predicts "Healthy" for everyone achieves 99% accuracy. However, this model is completely useless because it misses 100% of cancer cases.

### 7.4 Relevance of Recall and ROC-AUC in Healthcare
- **Recall (Sensitivity):** Quantifies the proportion of actual cancer patients correctly diagnosed. A False Negative (missing a cancer case) can lead to death, while a False Positive (false alarm) leads to anxiety and additional tests. Thus, high Recall is prioritized over Precision in medical screening.
- **ROC-AUC:** Measures the model's ability to rank cancer cases higher than healthy cases across all classification thresholds, independent of class distribution. It allows clinical teams to adjust thresholds to achieve 100% Recall while monitoring the False Positive Rate.

---

## 8. Task 7: Analytical Questions

### Q1: Why is KNN called a lazy learning algorithm?
**Answer:**  
KNN is a **lazy learner (instance-based)** because it performs no training phase or model parameter estimation. When `fit` is called, the training data is simply stored. Distances and voting are computed during the prediction phase. This leads to a training time complexity of $\mathcal{O}(1)$ and an inference time complexity of $\mathcal{O}(n \cdot d)$, which is slow for large datasets.

### Q2: Why is feature scaling required in KNN?
**Answer:**  
KNN uses distance metrics (e.g. Euclidean distance) to find neighbors. Features with large scales (like `worst area` up to $2500$) will dominate features with small scales (like `smoothness` $\approx 0.1$). Scaling ensures all features contribute equally to the distance calculation.

### Q3: Explain heuristic K selection using the $\sqrt{n}$ rule.
**Answer:**  
Choosing $K = \sqrt{n_{\text{train}}}$ balances bias and variance. Small $K$ (e.g. $K=1$) overfits to noise (high variance), while large $K$ (e.g. $K=n$) underfits to the majority class (high bias). $\sqrt{n}$ serves as a robust starting point that is rounded to the nearest odd integer to prevent voting ties.

### Q4: Why is cross-validation more reliable than a single train-test split?
**Answer:**  
A single split can suffer from partitioning bias (easy or hard samples grouped together). Cross-validation averages performance across multiple folds, ensuring every data point is used for both training and validation, providing a low-variance estimate of model generalization.

### Q5: How does K affect the bias-variance trade-off?
**Answer:**  
- **Small $K$ (e.g. $K=1$):** High variance, low bias. The boundary is complex and fits noise.
- **Large $K$:** Low variance, high bias. The boundary is overly smoothed, ignoring local data structures.

### Q6: Why is recall more important than accuracy in cancer prediction?
**Answer:**  
Accuracy does not penalize False Negatives (FN) differently than False Positives (FP). In cancer prediction, a False Negative (missing cancer) can be fatal, making Recall ($\frac{\text{TP}}{\text{TP} + \text{FN}}$) the primary metric to optimize.

### Q7: What is the limitation of very large K values?
**Answer:**  
As $K \rightarrow n$, the model suffers from high bias, eventually predicting the majority class for all samples. It increases computational search times during inference and loses all boundary details.

---

## 9. Conclusion
1. **Optimal $K$:** The heuristic suggested $K=21$, and 10-Fold CV identified **$K=11$** as optimal, achieving a mean validation accuracy of **$97.01\%$**.
2. **Splits:** The 80:20 split provided a robust balance between training representation and test stability.
3. **Model Performance:** The final model ($K=11$) achieved **$97.37\%$ test accuracy**, **$100\%$ Recall for benign cases**, and an **AUC of $0.9922$**.
4. **Key Insight:** Continuous evaluation (Lab 3) measures error *magnitude* to fit physical quantities, while classification (Lab 4) evaluates decision *correctness* to separate categories. In medical domains, Recall and ROC-AUC are critical because they account for the high cost of False Negatives.
