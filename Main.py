import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, GlobalMaxPooling1D
from tensorflow.keras.optimizers import Adam


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

# Feature Extraction
def feature_extraction(X):
    extracted_features = pd.DataFrame()
    extracted_features['age'] = X['age']
    extracted_features['sex'] = X['sex']
    extracted_features['cp'] = X['cp']
    extracted_features['restecg'] = X['restecg']
    extracted_features['thalach'] = X['thalach']
    extracted_features['pulse_pressure'] = X['trestbps'] - X['oldpeak']
    extracted_features['stress_index'] = X['thalach'] / (X['trestbps'] + 1e-5)
    extracted_features['age_normalized_chol'] = X['chol'] / (X['age'] + 1e-5)
    extracted_features['interaction_cp_thalach'] = X['cp'] * X['thalach']
    return extracted_features

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

# Define AEHOM Class
class AEHOM:
    def __init__(self, num_clans, num_features, lower_bounds, upper_bounds):
        self.num_clans = num_clans
        self.num_features = num_features
        self.lower_bounds = np.array(lower_bounds)
        self.upper_bounds = np.array(upper_bounds)

    def initialize_population(self, population_size):
        return np.random.uniform(self.lower_bounds, self.upper_bounds, (population_size, self.num_features))

    def evaluate_fitness(self, solution, model_builder, X_train, y_train, X_val, y_val):
        num_filters = int(solution[0])
        learning_rate = solution[1]
        dropout_rate = solution[2]

        model = model_builder(num_filters, learning_rate, dropout_rate)
        X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_val_reshaped = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)

        history = model.fit(
            X_train_reshaped,
            y_train,
            epochs=10,  # Short training for fitness evaluation
            batch_size=32,
            verbose=0,
            validation_data=(X_val_reshaped, y_val),
        )
        return max(history.history['val_accuracy'])

    def optimize(self, model_builder, X_train, y_train, X_val, y_val, population_size=10, max_iterations=20):
        population = self.initialize_population(population_size)
        best_solution = None
        best_fitness = float('-inf')

        for iteration in range(max_iterations):
            fitness_scores = [
                self.evaluate_fitness(sol, model_builder, X_train, y_train, X_val, y_val)
                for sol in population
            ]

            iteration_best_fitness = max(fitness_scores)
            iteration_best_solution = population[np.argmax(fitness_scores)]

            if iteration_best_fitness > best_fitness:
                best_fitness = iteration_best_fitness
                best_solution = iteration_best_solution

            for i in range(len(population)):
                if i != np.argmax(fitness_scores):  # Skip the best solution
                    population[i] = (
                        iteration_best_solution
                        + np.random.uniform(-0.1, 0.1) * (iteration_best_solution - population[i])
                    )
                    population[i] = np.clip(population[i], self.lower_bounds, self.upper_bounds)

            print(f"Iteration {iteration + 1}/{max_iterations}, Best Fitness: {best_fitness}")

        return best_solution, best_fitness


# Define MLDCNN Builder
def build_mldcnn_with_params(num_filters, learning_rate, dropout_rate):
    model = Sequential()
    model.add(Conv1D(filters=num_filters, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=num_filters * 2, kernel_size=3, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Apply AEHOM Optimization
X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train_balanced, y_train_balanced, test_size=0.2, random_state=42)

aehom = AEHOM(num_clans=5, num_features=3, lower_bounds=[32, 0.0001, 0.2], upper_bounds=[128, 0.01, 0.5])
best_solution, best_fitness = aehom.optimize(
    model_builder=build_mldcnn_with_params,
    X_train=X_train_sub,
    y_train=y_train_sub,
    X_val=X_val,
    y_val=y_val,
    population_size=10,
    max_iterations=10
)

print("\nBest Hyperparameters Found:")
print(f"Number of Filters: {int(best_solution[0])}")
print(f"Learning Rate: {best_solution[1]:.6f}")
print(f"Dropout Rate: {best_solution[2]:.2f}")
print(f"Best Validation Accuracy: {best_fitness:.2f}")


# Train Final Model with Optimized Hyperparameters
num_filters = int(best_solution[0])
learning_rate = best_solution[1]
dropout_rate = best_solution[2]

final_model = build_mldcnn_with_params(num_filters, learning_rate, dropout_rate)
X_train_final_reshaped = X_train_balanced.reshape(X_train_balanced.shape[0], X_train_balanced.shape[1], 1)
X_test_final_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

history = final_model.fit(
    X_train_final_reshaped,
    y_train_balanced,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

loss, accuracy = final_model.evaluate(X_test_final_reshaped, y_test, verbose=0)
print(f"\nTest Accuracy with Optimized Hyperparameters: {accuracy:.2f}")

# Final Evaluation
y_pred = (final_model.predict(X_test_final_reshaped) > 0.5).astype("int32")
print("\nFinal Classification Report:")
print(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

y_pred_proba = final_model.predict(X_test_final_reshaped).ravel()
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()