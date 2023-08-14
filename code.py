# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load and preprocess data (replace with your actual data loading code)
data = pd.read_csv('flower_and_tumor_data.csv')
X = data.drop(columns=['target_column'])  # Features
y = data['target_column']  # Labels

# Split data into training, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Train a Support Vector Machine (SVM) classifier
classifier = SVC(kernel='linear', C=1)
classifier.fit(X_train_scaled, y_train)

# Make predictions on validation set
y_val_pred = classifier.predict(X_val_scaled)

# Evaluate the model on validation set
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {val_accuracy:.2f}")

# Make predictions on test set
y_test_pred = classifier.predict(X_test_scaled)

# Evaluate the model on test set
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Print detailed classification report
class_report = classification_report(y_test, y_test_pred)
print("Classification Report:\n", class_report)
