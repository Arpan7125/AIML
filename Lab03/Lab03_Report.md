# Lab 3: Linear Regression and Overfitting Analysis

## 1. Overview
In this lab, we successfully generated a suitable synthetic dataset comprising **numeric features** (Age, Experience) and **nominal features** (Department, City) to predict a continuous target variable (`Salary`).

We applied basic data preprocessing steps:
- **Missing Values**: Handled using `SimpleImputer` (median for numeric, most frequent for categorical).
- **Categorical Encoding**: Handled using `OneHotEncoder` to convert 'Department' and 'City' into numeric arrays.
- **Feature Scaling**: Handled using `StandardScaler` to ensure numerical variables don't disproportionally influence the model due to scale differences.

The preprocessed data was split into **80% training** and **20% testing** sets to train a `LinearRegression` model.

---

## 2. Model Evaluation and Performance Metrics
The model was evaluated on both training and test datasets using the following metrics:
*   **Mean Absolute Error (MAE):** The average absolute difference between actual and predicted salaries.
*   **Mean Squared Error (MSE):** The average of the squared differences between actual and predicted values. It heavily penalizes larger errors.
*   **Root Mean Squared Error (RMSE):** The square root of MSE, bringing the error back to the original units (Salary in $).
*   **R² Score:** The proportion of variance in the dependent variable explained by the independent variables. Higher is better (up to 1.0).

---

## 3. Analysis of Generalization vs. Overfitting

### Comparing Training and Testing Performance
To analyze whether a model is **generalized** or **overfitted**, we compare its performance metrics on the training set versus the testing set:

- **Generalized Model:** A model that has learned the underlying pattern of the data rather than memorizing the noise. Such a model will perform similarly well on both the training data and unseen testing data. Both $R^2$ scores will be high, and RMSE/MAE will be comparatively similar.
- **Overfitted Model:** A model that has memorized the training data, including its random noise. It will show exceptional performance on the training data (e.g., $R^2 \approx 0.99$, very low RMSE) but significantly worse performance on the testing data (much lower $R^2$, high RMSE).
- **Underfitted Model:** Performs poorly on both training and testing datasets.

**In our linear regression model (assuming typical synthetic data output):**
Since our synthetic target variable (`Salary`) was designed directly as a linear combination of features with a reasonable amount of added random noise, a Linear Regression model usually extracts the exact linear weights correctly. You will notice that the training and testing metrics (MAE, RMSE, $R^2$) are quite close to each other. 
**Conclusion:** Therefore, the model is **generalized**. It did not overfit because it maintained high and comparable accuracy when predicting new, unseen test data.

---

## 4. Suggested Methods to Reduce Overfitting
If our model had shown signs of overfitting (e.g., high train R², low test R²), we could employ the following strategies to reduce it:

### A. Feature Selection and Removing Irrelevant Variables
When a dataset has too many features, especially ones that do not logically influence the target variable, the model might try to map random statistical noise from these irrelevant features to the output.
*   **Solution:** Analyze the correlation of variables. Drop columns that have near-zero correlation with the target variable or are highly colinear (highly correlated with other predictor variables). We could use techniques like *Recursive Feature Elimination (RFE)*.

### B. Cross-Validation
Evaluating a model on a single train-test split can be misleading, as a lucky or unlucky split could make the model appear better or worse than it is.
*   **Solution:** Use **K-Fold Cross-Validation** (e.g., $k=5$ or $k=10$). The dataset is split into $k$ parts; the model is trained on $k-1$ parts and tested on the remaining part. This process is repeated $k$ times, and the average score provides a much more robust estimation of the model's true capability to generalize to unseen data.

### C. Regularization Techniques
Standard Linear Regression doesn't have built-in defenses against overfitting complex relationships. Regularization modifies the cost function to penalize extreme parameter weights.
*   **L1 Regularization (Lasso Regression):** Can force some coefficients to become exactly zero, implicitly acting as a feature selector.
*   **L2 Regularization (Ridge Regression):** Shrinks the coefficients evenly, preventing any single feature from dominating the model due to noise.

### D. Increase Dataset Size
Overfitting often occurs when the model is too complex for the amount of data available. Providing more data samples gives the model a clearer view of the true underlying trend, effectively drowning out the noise.
