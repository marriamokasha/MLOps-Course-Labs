"""
This module contains functions to preprocess and train the model
for bank consumer churn prediction.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder,  StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

### Import MLflow
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import joblib
import os

def rebalance(data):
    """
    Resample data to keep balance between target classes.

    The function uses the resample function to downsample the majority class to match the minority class.

    Args:
        data (pd.DataFrame): DataFrame

    Returns:
        pd.DataFrame): balanced DataFrame
    """
    churn_0 = data[data["Exited"] == 0]
    churn_1 = data[data["Exited"] == 1]
    if len(churn_0) > len(churn_1):
        churn_maj = churn_0
        churn_min = churn_1
    else:
        churn_maj = churn_1
        churn_min = churn_0
    churn_maj_downsample = resample(
        churn_maj, n_samples=len(churn_min), replace=False, random_state=1234
    )

    return pd.concat([churn_maj_downsample, churn_min])


def preprocess(df):
    """
    Preprocess and split data into training and test sets.

    Args:
        df (pd.DataFrame): DataFrame with features and target variables

    Returns:
        ColumnTransformer: ColumnTransformer with scalers and encoders
        pd.DataFrame: training set with transformed features
        pd.DataFrame: test set with transformed features
        pd.Series: training set target
        pd.Series: test set target
    """
    filter_feat = [
        "CreditScore",
        "Geography",
        "Gender",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
        "Exited",
    ]
    cat_cols = ["Geography", "Gender"]
    num_cols = [
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
    ]
    data = df.loc[:, filter_feat]
    data_bal = rebalance(data=data)
    X = data_bal.drop("Exited", axis=1)
    y = data_bal["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1912
    )
    col_transf = make_column_transformer(
        (StandardScaler(), num_cols), 
        (OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols),
        remainder="passthrough",
    )

    X_train = col_transf.fit_transform(X_train)
    X_train = pd.DataFrame(X_train, columns=col_transf.get_feature_names_out())

    X_test = col_transf.transform(X_test)
    X_test = pd.DataFrame(X_test, columns=col_transf.get_feature_names_out())

    # Log the transformer as an artifact

    return col_transf, X_train, X_test, y_train, y_test


def train_and_evaluate_model(model_name, model, X_train, X_test, y_train, y_test):
    """
    Train a model and evaluate its performance.

    Args:
        model_name (str): Name of the model
        model: Model instance to train
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Test features
        y_train (pd.Series): Training target
        y_test (pd.Series): Test target

    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Start a new run for this model
    with mlflow.start_run(nested=True, run_name=model_name) as run:
        print(f"\nTraining {model_name}...")
        
        # Log model parameters
        params = model.get_params()
        mlflow.log_params(params)
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Log metrics
        mlflow.log_metrics({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        })
        
        # Create and log confusion matrix
        conf_mat = confusion_matrix(y_test, y_pred, labels=model.classes_)
        conf_mat_disp = ConfusionMatrixDisplay(
            confusion_matrix=conf_mat, display_labels=model.classes_
        )
        fig, ax = plt.subplots(figsize=(8, 6))
        conf_mat_disp.plot(ax=ax)
        plt.title(f"Confusion Matrix - {model_name}")
        
        # Save confusion matrix plot locally
        cm_path = f"confusion_matrix_{model_name}.png"
        plt.savefig(cm_path)
        
        # Log the plot as an artifact
        mlflow.log_artifact(cm_path)
        
        # Log model with signature
        signature = infer_signature(X_train, y_pred)
        mlflow.sklearn.log_model(model, f"{model_name}_model", signature=signature)
        
        # Log tags
        mlflow.set_tags({
            "model_type": model_name,
            "data_version": "v1.0"
        })
        
        # Print metrics
        print(f"{model_name} Metrics:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        
        # Delete the saved image after logging
        if os.path.exists(cm_path):
            os.remove(cm_path)
            
        return {
            "model": model,
            "metrics": {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            },
            "run_id": run.info.run_id
        }
def save_preprocessor(preprocessor, file_path="models/preprocessor.pkl"):
    """
    Save the preprocessing transformer to disk.
    
    Args:
        preprocessor: Fitted column transformer
        file_path: Path where to save the transformer
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Save the preprocessor using joblib
    joblib.dump(preprocessor, file_path)
    print(f"Preprocessor saved to {file_path}")
    
    return file_path


def save_best_model(model, model_name, file_path="models/best_model.pkl"):
    """
    Save the best model to disk.
    
    Args:
        model: Trained model
        model_name: Name of the model
        file_path: Path where to save the model
        
    Returns:
        str: Path to the saved model
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Save model name and object as a dictionary
    model_info = {
        "model_name": model_name,
        "model_object": model
    }
    
    # Save using joblib
    joblib.dump(model_info, file_path)
    print(f"Best model ({model_name}) saved to {file_path}")
    
    return file_path


def load_preprocessor(file_path="models/preprocessor.pkl"):
    """
    Load the preprocessing transformer from disk.
    
    Args:
        file_path: Path to the saved preprocessor
        
    Returns:
        The loaded preprocessor
    """
    return joblib.load(file_path)


def load_best_model(file_path="models/best_model.pkl"):
    """
    Load the best model from disk.
    
    Args:
        file_path: Path to the saved model
        
    Returns:
        dict: Dictionary containing model name and model object
    """
    return joblib.load(file_path)

def main():
    # Set the tracking URI for MLflow
    mlflow.set_tracking_uri("http://localhost:5000")  # Change if using a different URI

    # Set the experiment name
    experiment_name = "bank_churn_prediction"
    mlflow.set_experiment(experiment_name)

    # Start a new run for the overall experiment
    with mlflow.start_run(run_name="multi_model_comparison") as parent_run:
        print("Starting bank customer churn prediction experiment...")

        # Log experiment-level tags
        mlflow.set_tags({
            "experiment_type": "model_comparison",
            "data_source": "bank_customer_churn",
            "purpose": "churn_prediction"
        })
        
        # Load and preprocess data
        df = pd.read_csv("dataset/Churn_Modelling.csv")
        col_transf, X_train, X_test, y_train, y_test = preprocess(df)
        
        # Save the preprocessor to disk and log its path as an artifact in MLflow
        preprocessor_path = save_preprocessor(col_transf)
        mlflow.log_artifact(preprocessor_path, "preprocessor")
        
        # Define models to train and evaluate
        models = {
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
            "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        
        # Train and evaluate each model
        results = {}
        for model_name, model in models.items():
            results[model_name] = train_and_evaluate_model(
                model_name, model, X_train, X_test, y_train, y_test
            )
        
        # Log data samples
        mlflow.log_artifact("dataset/Churn_Modelling.csv", "data_samples")
        
        # Compare and find the best model
        best_model_name = max(results.items(), key=lambda x: x[1]["metrics"]["f1_score"])[0]
        best_model_obj = results[best_model_name]["model"]
        best_model_metrics = results[best_model_name]["metrics"]
        best_run_id = results[best_model_name]["run_id"]
        
        # Save the best model to disk and log its path as an artifact in MLflow
        best_model_path = save_best_model(best_model_obj, best_model_name)
        mlflow.log_artifact(best_model_path, "best_model")
        
        # Log a simple text file with the best model information
        best_model_info = (
            f"Best Model: {best_model_name}\n"
            f"F1 Score: {best_model_metrics['f1_score']:.4f}\n"
            f"Accuracy: {best_model_metrics['accuracy']:.4f}\n"
            f"Precision: {best_model_metrics['precision']:.4f}\n"
            f"Recall: {best_model_metrics['recall']:.4f}\n"
            f"Run ID: {best_run_id}\n"
        )
        
        with open("best_model_info.txt", "w") as f:
            f.write(best_model_info)
        
        mlflow.log_artifact("best_model_info.txt")
        
        # Print best model information
        print(f"\nBest model based on F1 score: {best_model_name}")
        print(f"F1 score: {best_model_metrics['f1_score']:.4f}")
        print(f"Run ID: {best_run_id}")
        print(f"Preprocessor saved to: {preprocessor_path}")
        print(f"Best model saved to: {best_model_path}")



if __name__ == "__main__":
    main()
    