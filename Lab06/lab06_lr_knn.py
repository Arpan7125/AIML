# =============================================================================
# Lab 06 – Logistic Regression vs K-Nearest Neighbours (KNN)
# Dataset : Breast Cancer Wisconsin (Diagnostic)
# Source  : sklearn.datasets  (built-in, no manual download needed)
# =============================================================================

# ── Imports ───────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer          # built-in dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)


# =============================================================================
# STEP 1 – Load and Explore the Dataset
# =============================================================================
def step1_load():
    print("=" * 60)
    print("STEP 1 : Load & Explore Dataset")
    print("=" * 60)

    # load_breast_cancer() returns a Bunch object (like a dict)
    data = load_breast_cancer()

    # Convert to a DataFrame so we can inspect it easily
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target          # 0 = malignant, 1 = benign

    print(f"\nDataset shape : {df.shape}  (rows, columns)")
    print(f"Classes       : {data.target_names}")   # ['malignant' 'benign']
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nClass distribution:\n{df['target'].value_counts()}")

    return df, data


# =============================================================================
# STEP 2 – Preprocessing
# =============================================================================
def step2_preprocess(df, data):
    print("\n" + "=" * 60)
    print("STEP 2 : Preprocessing")
    print("=" * 60)

    # 2a. Check for missing values
    print(f"\nMissing values per column:\n{df.isnull().sum()}")
    # This dataset has NO missing values, but we always check

    # 2b. Separate features (X) and target label (y)
    X = df.drop('target', axis=1)   # all 30 feature columns
    y = df['target']                # 0 = malignant, 1 = benign

    # 2c. Train / Test split  (80 % train, 20 % test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    # stratify=y keeps the same class ratio in both splits

    # 2d. Feature Scaling – very important for KNN and Logistic Regression
    #     StandardScaler makes every feature have mean=0 and std=1
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)   # learn + apply on train
    X_test_sc  = scaler.transform(X_test)        # only apply on test (no leakage)

    print(f"\nTraining samples : {X_train_sc.shape[0]}")
    print(f"Testing  samples : {X_test_sc.shape[0]}")

    return X_train_sc, X_test_sc, y_train, y_test


# =============================================================================
# STEP 3 – Train Models
# =============================================================================
def step3_train(X_train, y_train):
    print("\n" + "=" * 60)
    print("STEP 3 : Train Classifiers")
    print("=" * 60)

    # ── Logistic Regression ──────────────────────────────────────────────────
    # max_iter=1000  → allow more iterations so it always converges
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    print("\n[OK] Logistic Regression trained successfully")

    # ── K-Nearest Neighbours ─────────────────────────────────────────────────
    # n_neighbors=5  → look at the 5 closest neighbours and vote
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    print("[OK] KNN (k=5) trained successfully")

    return lr_model, knn_model


# =============================================================================
# STEP 4 – Evaluate Models
# =============================================================================
def evaluate(name, model, X_test, y_test):
    """Print all classification metrics for one model."""
    y_pred = model.predict(X_test)

    acc  = accuracy_score (y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score   (y_test, y_pred)
    f1   = f1_score       (y_test, y_pred)
    cm   = confusion_matrix(y_test, y_pred)

    print(f"\n" + "-" * 40)
    print(f"  Model : {name}")
    print("-" * 40)
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f} %)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"  {cm}")
    print(f"  (rows = Actual, cols = Predicted)")

    return acc, prec, rec, f1


def step4_evaluate(lr_model, knn_model, X_test, y_test):
    print("\n" + "=" * 60)
    print("STEP 4 : Evaluate Both Classifiers")
    print("=" * 60)

    lr_metrics  = evaluate("Logistic Regression", lr_model,  X_test, y_test)
    knn_metrics = evaluate("KNN (k=5)",           knn_model, X_test, y_test)

    return lr_metrics, knn_metrics


# =============================================================================
# STEP 5 – Comparison Table
# =============================================================================
def step5_compare(lr_metrics, knn_metrics):
    print("\n" + "=" * 60)
    print("STEP 5 : Comparison Table")
    print("=" * 60)

    metrics_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]

    comparison = pd.DataFrame({
        "Metric"               : metrics_labels,
        "Logistic Regression"  : [f"{v:.4f}" for v in lr_metrics],
        "KNN (k=5)"            : [f"{v:.4f}" for v in knn_metrics],
    })

    print(f"\n{comparison.to_string(index=False)}")


# =============================================================================
# STEP 6 – Conclusion
# =============================================================================
def step6_conclusion(lr_metrics, knn_metrics):
    print("\n" + "=" * 60)
    print("STEP 6 : Conclusion")
    print("=" * 60)

    lr_acc, _, _, lr_f1   = lr_metrics
    knn_acc, _, _, knn_f1 = knn_metrics

    print("\nInterpretation:")
    winner = "Logistic Regression" if lr_f1 >= knn_f1 else "KNN (k=5)"

    print(f"  Logistic Regression F1 : {lr_f1:.4f}")
    print(f"  KNN F1                 : {knn_f1:.4f}")
    print(f"\n  Better classifier for this dataset : {winner}")
    print("""
  Reason :
  - Logistic Regression works well when the data is (approximately)
    linearly separable and features are scaled.
  - KNN is a distance-based method; it is sensitive to feature scale
    and the choice of k.
  - On the Breast Cancer dataset, Logistic Regression typically achieves
    slightly higher precision and recall because it creates a smooth
    decision boundary, reducing false negatives (missed cancer cases).
  - In medical diagnosis, Recall is especially important — a missed
    malignant case (false negative) is more dangerous than a false alarm.
    """)


# =============================================================================
# MAIN
# =============================================================================
def main():
    df, data                    = step1_load()
    X_train, X_test, y_train, \
    y_test                      = step2_preprocess(df, data)
    lr_model, knn_model         = step3_train(X_train, y_train)
    lr_metrics, knn_metrics     = step4_evaluate(lr_model, knn_model,
                                                  X_test, y_test)
    step5_compare(lr_metrics, knn_metrics)
    step6_conclusion(lr_metrics, knn_metrics)


if __name__ == "__main__":
    main()
