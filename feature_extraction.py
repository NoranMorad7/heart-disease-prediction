import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

# Define feature extraction functions
def calculate_vmax(v):
    return np.max(v, axis=1)

def calculate_thd(hc, pff):
    return np.sum(hc * pff, axis=1)

def calculate_hr(rr, t):
    return 60 * rr / t

def calculate_zcr(sc, snc):
    return sc / (sc + snc)

def calculate_entropy(pxy):
    return -np.sum(pxy * np.log(pxy), axis=1)

def calculate_energy(pxy):
    return np.sum(pxy ** 2, axis=1)

def calculate_sd(rr):
    return np.std(rr, axis=1)

def calculate_k(rr):
    return np.mean(np.diff(rr, axis=1), axis=1)

def calculate_r(rr):
    return np.sqrt(np.mean(np.diff(rr, axis=1) ** 2, axis=1))

def feature_extraction(X):
    """
    Perform feature extraction based on available columns and domain knowledge.
    Args:
        X (pd.DataFrame): Input feature set containing raw signals and measurements.
    Returns:
        pd.DataFrame: Feature set with extracted and derived features.
    """
def feature_extraction(X):
    """
    Perform feature extraction based on available columns and domain knowledge.
    Handles missing columns dynamically.
    """
    extracted_features = pd.DataFrame()

    # Basic features (check for existence)
    if 'age' in X:
        extracted_features['age'] = X['age']
    if 'sex' in X:
        extracted_features['sex'] = X['sex']
    if 'cp' in X:
        extracted_features['cp'] = X['cp']
    if 'restecg' in X:
        extracted_features['restecg'] = X['restecg']
    if 'thalach' in X:
        extracted_features['thalach'] = X['thalach']

    # Derived features (check for required columns)
    if 'trestbps' in X and 'oldpeak' in X:
        extracted_features['pulse_pressure'] = X['trestbps'] - X['oldpeak']
    if 'thalach' in X and 'trestbps' in X:
        extracted_features['stress_index'] = X['thalach'] / (X['trestbps'] + 1e-5)  # Avoid division by zero
    if 'chol' in X and 'age' in X:
        extracted_features['age_normalized_chol'] = X['chol'] / (X['age'] + 1e-5)
    if 'cp' in X and 'thalach' in X:
        extracted_features['interaction_cp_thalach'] = X['cp'] * X['thalach']
    if 'thal' in X and 'thalach' in X:
        extracted_features['interaction_thal_thalach'] = X['thal'] * X['thalach']

    return extracted_features
