import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def main():
    print("=== Step 1: Dataset Generation ===")
    # 1. Create a suitable dataset with numeric and nominal features, including missing values
    np.random.seed(42)
    data = {
        'Age': np.random.randint(22, 60, 200),
        'Experience': np.random.randint(1, 35, 200),
        'Department': np.random.choice(['IT', 'HR', 'Finance', 'Marketing'], 200),
        'City': np.random.choice(['New York', 'London', 'Paris', 'Tokyo'], 200)
    }

    df = pd.DataFrame(data)

    # Calculate Target (Salary) based on a linear combination with some noise
    dept_bonus = {'IT': 10000, 'Finance': 12000, 'Marketing': 8000, 'HR': 6000}
    city_bonus = {'New York': 20000, 'London': 15000, 'Tokyo': 10000, 'Paris': 8000}

    df['Salary'] = (df['Age'] * 500 + 
                    df['Experience'] * 1500 + 
                    df['Department'].map(dept_bonus) + 
                    df['City'].map(city_bonus) + 
                    np.random.normal(0, 5000, 200)) # adding random noise

    # Introduce some missing values to satisfy preprocessing requirements
    df.loc[5:20, 'Age'] = np.nan
    df.loc[30:45, 'City'] = np.nan

    print("Sample dataset head:\n", df.head())
    print("\nMissing values before preprocessing:\n", df.isnull().sum())

    # 2. Perform basic data preprocessing
    print("\n=== Step 2: Data Preprocessing ===")
    X = df.drop('Salary', axis=1)
    y = df['Salary']

    numeric_features = ['Age', 'Experience']
    categorical_features = ['Department', 'City']

    # Pipeline for numeric features: Impute missing values with median, then scale
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Pipeline for categorical features: Impute missing with most frequent, then one-hot encode
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine preprocessing steps using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # 3. Split dataset into training and testing sets
    print("Splitting data into 80% training and 20% testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apply transformations
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    # 4. Build a Linear Regression model
    print("\n=== Step 3: Building and Training Linear Regression Model ===")
    model = LinearRegression()
    model.fit(X_train_preprocessed, y_train)
    print("Model training complete.")

    # Predictions
    y_train_pred = model.predict(X_train_preprocessed)
    y_test_pred = model.predict(X_test_preprocessed)

    # 5. Evaluate the model
    print("\n=== Step 4: Model Evaluation ===")
    def evaluate_model(y_true, y_pred, dataset_name):
        print(f"--- Metrics for {dataset_name} Data ---")
        print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_true, y_pred):.2f}")
        print(f"Mean Squared Error (MSE): {mean_squared_error(y_true, y_pred):.2f}")
        print(f"Root Mean Squared Error (RMSE): {np.sqrt(mean_squared_error(y_true, y_pred)):.2f}")
        print(f"R² Score: {r2_score(y_true, y_pred):.4f}\n")

    evaluate_model(y_train, y_train_pred, "Training")
    evaluate_model(y_test, y_test_pred, "Testing")

if __name__ == "__main__":
    main()
