"""
Model training and evaluation module for Titanic survival prediction.

This module provides functions to train machine learning models,
evaluate their performance, and save/load trained models.
"""

import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = "random_forest",
    **model_params
) -> Any:
    """
    Train a classification model on the provided data.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix for training.
    y : pd.Series
        Target variable (survival labels).
    model_type : str, optional
        Type of model to train. Options: 'random_forest', 'logistic_regression'.
        Default is 'random_forest'.
    **model_params
        Additional keyword arguments to pass to the model constructor.
        
    Returns
    -------
    model
        Trained scikit-learn model object.
        
    Examples
    --------
    >>> model = train_model(X_train, y_train, model_type='random_forest', n_estimators=100)
    >>> model = train_model(X_train, y_train, model_type='logistic_regression', max_iter=1000)
    """
    # Set default parameters if none provided
    if model_type == "random_forest":
        default_params = {
            "n_estimators": 100,
            "max_depth": 5,
            "random_state": 42
        }
        default_params.update(model_params)
        model = RandomForestClassifier(**default_params)
        
    elif model_type == "logistic_regression":
        default_params = {
            "max_iter": 1000,
            "random_state": 42
        }
        default_params.update(model_params)
        model = LogisticRegression(**default_params)
        
    else:
        raise ValueError(
            f"Unknown model_type: {model_type}. "
            "Choose 'random_forest' or 'logistic_regression'."
        )
    
    # Train the model
    model.fit(X, y)
    
    print(f"✓ {model_type.replace('_', ' ').title()} model trained successfully")
    print(f"  Training samples: {len(X)}")
    print(f"  Features: {X.shape[1]}")
    
    return model


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Evaluate a trained model on test data and return performance metrics.
    
    Parameters
    ----------
    model
        Trained scikit-learn model object.
    X_test : pd.DataFrame
        Feature matrix for testing.
    y_test : pd.Series
        True labels for test set.
    verbose : bool, optional
        If True, print detailed evaluation results. Default is True.
        
    Returns
    -------
    Dict[str, float]
        Dictionary containing evaluation metrics:
        - 'accuracy': Overall accuracy score
        - 'precision': Precision score
        - 'recall': Recall score
        - 'f1': F1 score
        
    Examples
    --------
    >>> metrics = evaluate_model(model, X_test, y_test)
    >>> print(f"Accuracy: {metrics['accuracy']:.3f}")
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0)
    }
    
    if verbose:
        print("=" * 50)
        print("MODEL EVALUATION RESULTS")
        print("=" * 50)
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1']:.4f}")
        print("\n" + "=" * 50)
        print("CONFUSION MATRIX")
        print("=" * 50)
        cm = confusion_matrix(y_test, y_pred)
        print(f"True Negatives:  {cm[0, 0]}")
        print(f"False Positives: {cm[0, 1]}")
        print(f"False Negatives: {cm[1, 0]}")
        print(f"True Positives:  {cm[1, 1]}")
        print("\n" + "=" * 50)
        print("CLASSIFICATION REPORT")
        print("=" * 50)
        print(classification_report(y_test, y_pred, target_names=['Died', 'Survived']))
    
    return metrics


def save_model(model: Any, filepath: str = "models/trained_model.pkl") -> None:
    """
    Save a trained model to disk using pickle.
    
    Parameters
    ----------
    model
        Trained scikit-learn model object to save.
    filepath : str, optional
        Path where the model should be saved. Default is 'models/trained_model.pkl'.
        
    Examples
    --------
    >>> save_model(model, "models/random_forest_model.pkl")
    """
    # Create directory if it doesn't exist
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    # Save model
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"✓ Model saved to: {filepath}")


def load_model(filepath: str = "models/trained_model.pkl") -> Any:
    """
    Load a trained model from disk.
    
    Parameters
    ----------
    filepath : str, optional
        Path to the saved model file. Default is 'models/trained_model.pkl'.
        
    Returns
    -------
    model
        Loaded scikit-learn model object.
        
    Examples
    --------
    >>> model = load_model("models/random_forest_model.pkl")
    >>> predictions = model.predict(X_test)
    """
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    
    print(f"✓ Model loaded from: {filepath}")
    return model


def compare_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> pd.DataFrame:
    """
    Train and compare multiple models on the same dataset.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training target variable.
    X_test : pd.DataFrame
        Testing feature matrix.
    y_test : pd.Series
        Testing target variable.
        
    Returns
    -------
    pd.DataFrame
        Comparison table with metrics for each model.
        
    Examples
    --------
    >>> comparison = compare_models(X_train, y_train, X_test, y_test)
    >>> print(comparison)
    """
    results = []
    
    model_configs = [
        ("Logistic Regression", "logistic_regression", {}),
        ("Random Forest", "random_forest", {"n_estimators": 100}),
        ("Random Forest (Deep)", "random_forest", {"n_estimators": 200, "max_depth": 10})
    ]
    
    for name, model_type, params in model_configs:
        print(f"\nTraining {name}...")
        model = train_model(X_train, y_train, model_type=model_type, **params)
        metrics = evaluate_model(model, X_test, y_test, verbose=False)
        metrics['model'] = name
        results.append(metrics)
    
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df[['model', 'accuracy', 'precision', 'recall', 'f1']]
    
    return comparison_df
