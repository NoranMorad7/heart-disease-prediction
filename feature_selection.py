import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error
import random

# Define your working directory and file paths dynamically
working_directory = r"C:\Users\CONNECT\Downloads\Heart Disease Prediction"
data_file = os.path.join(working_directory, "Heart.csv")
output_file_path = os.path.join(working_directory, "selected_features.csv")

# Load the dataset
data = pd.read_csv(data_file)

# View the first few rows and column names
print("Dataset Preview:\n", data.head())
print("Column Names:\n", data.columns)
print("Size before removal of any duplicate values:", data.shape)

# Check for missing values
print("Missing values before substitution:\n", data.isnull().sum())

# 1. Handle Missing Values (Replace '?' with NaN and handle them)
data.replace('?', np.nan, inplace=True)  # Replace '?' with NaN
print("Missing values after replacement:\n", data.isnull().sum())

# Handle missing values: For numerical columns, replace NaN with the mean
numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
data[numerical_cols] = data[numerical_cols].apply(pd.to_numeric, errors='coerce')  # Convert any remaining non-numeric values to NaN
data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].mean())  # Replace NaNs with the mean

# For categorical columns, replace NaN with the mode (most frequent value)
categorical_cols = data.select_dtypes(include=['object', 'category']).columns
data[categorical_cols] = data[categorical_cols].fillna(data[categorical_cols].mode().iloc[0])  # Replace NaNs with the mode

# Verify that all missing values have been handled
print("Missing values after substitution:\n", data.isnull().sum())

# 2. Check for duplicates
duplicates = data.duplicated()
num_duplicates = duplicates.sum()
if num_duplicates > 0:
    print(f"Number of duplicate rows: {num_duplicates}")
    print("Duplicate rows:")
    print(data[duplicates])
else:
    print("No duplicate rows found.")

# 3. Segregation (Categorization)
# Optional: Categorize patients based on chest pain type ('cp')
# Map 'cp' column values to descriptive categories (if needed)

# 4. Focus on 'condition' (heart disease)
# Patients with condition = 1 (has heart disease) and 0 (no heart disease)
if 'num' in data.columns:
    heart_disease_data = data[data['num'] == 1]
    no_heart_disease_data = data[data['num'] == 0]
    print(f"Heart disease cases: {heart_disease_data.shape[0]}")
    print(f"No heart disease cases: {no_heart_disease_data.shape[0]}")

# 5. Display Preprocessed Data
print("Preprocessed Data:\n", data.head())

# Segregate features (X) and target (y)
X = data.drop(columns=['num'])
y = data['num']

# Standardize numerical features
scaler = StandardScaler()
numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Hyperparameters
DESIRED_FEATURES = 11  # Desired number of features to retain
K_FOLDS = 5  # K-folds for cross-validation
POPULATION_SIZE = 80  # Number of genomes (solutions) per generation
GENERATIONS = 18  # Number of generations for GA
CROSSOVER_RATE = 0.05  # Crossover rate
MUTATION_RATE = 0.05  # Mutation rate


# GA Fitness Function (MSE)
def fitness_function(X_train, y_train, selected_features):
    """Calculate the mean squared error (MSE) based on the selected features."""
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train.iloc[:, selected_features], y_train)
    y_pred = model.predict(X_train.iloc[:, selected_features])
    mse = mean_squared_error(y_train, y_pred)
    return mse


# Apply RFEM to eliminate the least important features
def apply_rfe(X, y, n_features):
    """Apply RFE to eliminate the least important features."""
    model = GradientBoostingClassifier(random_state=42)
    rfe = RFE(estimator=model, n_features_to_select=n_features)
    rfe.fit(X, y)
    return rfe.support_


# Perform genetic algorithm to generate and evaluate genomes
def genetic_algorithm(X, y):
    """Genetic algorithm to find the optimal feature set."""
    # Initialize random population of genomes (binary representation of features)
    population = []
    num_features = X.shape[1]

    # Randomly initialize the population
    for _ in range(POPULATION_SIZE):
        genome = random.sample(range(num_features), num_features)  # Random genome
        population.append(genome)

    # Iterate over generations
    for generation in range(GENERATIONS):
        print(f"Generation {generation + 1}/{GENERATIONS}")

        # Evaluate fitness for each genome
        fitness_scores = []
        for genome in population:
            fitness = fitness_function(X, y, genome)
            fitness_scores.append(fitness)

        # Select the top 40 genomes based on fitness (lowest MSE)
        selected_population = [x for _, x in sorted(zip(fitness_scores, population))][:40]

        # Generate new population via crossover and mutation
        next_population = []

        # Crossover
        while len(next_population) < POPULATION_SIZE:
            parent1, parent2 = random.sample(selected_population, 2)
            if random.random() < CROSSOVER_RATE:
                crossover_point = random.randint(1, len(parent1) - 1)
                child = parent1[:crossover_point] + parent2[crossover_point:]
                next_population.append(child)
            else:
                next_population.append(parent1)

        # Mutation
        for i in range(len(next_population)):
            if random.random() < MUTATION_RATE:
                mutation_point = random.randint(0, len(next_population[i]) - 1)
                next_population[i][mutation_point] = random.choice(range(X.shape[1]))

        # Replace old population with new generation
        population = next_population

    # Return the best solution (genome) after all generations
    best_genome = population[0]
    return best_genome


# Apply the genetic algorithm to select the best genome
best_genome = genetic_algorithm(X, y)

# Apply RFEM on the best genome
final_selected_features = apply_rfe(X.iloc[:, best_genome], y, DESIRED_FEATURES)

# Output the final selected features
selected_feature_names = X.columns[final_selected_features]
print(f"Selected Features after GA + RFEM: {list(selected_feature_names)}")

# Create a DataFrame for the selected features
selected_data = X.iloc[:, final_selected_features]

# K-Fold Cross-validation to evaluate model performance
kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
mse_scores = []

# Perform k-fold cross-validation
for train_index, test_index in kf.split(selected_data):
    X_train, X_test = selected_data.iloc[train_index], selected_data.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse_scores.append(mean_squared_error(y_test, y_pred))

# Print the average MSE score from cross-validation
print(f"Average MSE from K-Fold Cross-Validation: {np.mean(mse_scores)}")

# Save the selected features and corresponding rows of the dataset to a CSV file
selected_data.to_csv(output_file_path, index=False)
print(f"Selected features data saved to {output_file_path}")

