import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.layers import GlobalMaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE

# Load Dataset
data = pd.read_csv('/content/the data.csv')

# Check Available Columns
print("Available Columns in Dataset:", data.columns)

# Preprocess the Dataset
selected_features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'ca']
target = 'target'

# Ensure only existing columns are selected
existing_features = [col for col in selected_features if col in data.columns]
X = data[existing_features]
y = data[target]

# Perform Feature Extraction
X_extracted = feature_extraction(X)
print("Extracted Features:")
print(X_extracted.head())

# Scale Extracted Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_extracted)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Handle Class Imbalance
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Define K-Fold Cross-Validation
n_splits = 5  # Number of folds
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Create a function to build the model (required for KerasClassifier)
def build_mldcnn_for_kfold():
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_scaled.shape[1], 1)))
    model.add(MaxPooling1D(pool_size=1))
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Initialize Metrics
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

# K-Fold Cross-Validation Loop
for fold, (train_idx, val_idx) in enumerate(kfold.split(X_scaled, y)):
    print(f"\nProcessing Fold {fold + 1}/{n_splits}...")
    
    # Split the data into training and validation sets
    X_train_fold, X_val_fold = X_scaled[train_idx], X_scaled[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

    # Handle Class Imbalance with SMOTE for the Training Fold
    smote = SMOTE(random_state=42)
    X_train_fold_balanced, y_train_fold_balanced = smote.fit_resample(X_train_fold, y_train_fold)

    # Reshape Data for Conv1D
    X_train_fold_reshaped = X_train_fold_balanced.reshape(X_train_fold_balanced.shape[0], X_train_fold_balanced.shape[1], 1)
    X_val_fold_reshaped = X_val_fold.reshape(X_val_fold.shape[0], X_val_fold.shape[1], 1)

    # Build and Train the Model
    model = build_mldcnn_for_kfold()
    history = model.fit(X_train_fold_reshaped, y_train_fold_balanced, epochs=50, batch_size=32, verbose=0)
    
    # Evaluate the Model
    y_val_pred = (model.predict(X_val_fold_reshaped) > 0.5).astype("int32")
    accuracy_scores.append(accuracy_score(y_val_fold, y_val_pred))
    precision_scores.append(precision_score(y_val_fold, y_val_pred))
    recall_scores.append(recall_score(y_val_fold, y_val_pred))
    f1_scores.append(f1_score(y_val_fold, y_val_pred))

    print(f"Fold Accuracy: {accuracy_scores[-1]:.2f}")
    print(f"Fold Precision: {precision_scores[-1]:.2f}")
    print(f"Fold Recall: {recall_scores[-1]:.2f}")
    print(f"Fold F1-Score: {f1_scores[-1]:.2f}")

# Aggregate and Display Final Metrics
print("\nFinal K-Fold Cross-Validation Results:")
print(f"Mean Accuracy: {np.mean(accuracy_scores):.2f} ± {np.std(accuracy_scores):.2f}")
print(f"Mean Precision: {np.mean(precision_scores):.2f} ± {np.std(precision_scores):.2f}")
print(f"Mean Recall: {np.mean(recall_scores):.2f} ± {np.std(recall_scores):.2f}")
print(f"Mean F1-Score: {np.mean(f1_scores):.2f} ± {np.std(f1_scores):.2f}")