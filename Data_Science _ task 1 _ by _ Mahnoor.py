import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset
data = pd.read_csv("https://drive.google.com/uc?export=download&id=1hc5mqZ3rBeJCTk0OEHjKGEf_U-41BZ4M")
print("Dataset shape:", data.shape)
print("Missing values after cleaning:", data.isnull().sum().sum())

# Drop unnecessary columns safely
cols_to_drop = [col for col in ["Name", "Ticket", "Cabin"] if col in data.columns]
data = data.drop(cols_to_drop, axis=1)

# Handle missing values
data = data.assign(
    Age=data["Age"].fillna(data["Age"].median()),
    Embarked=data["Embarked"].fillna(data["Embarked"].mode()[0])
)

# Encode categorical variables safely
le_sex = LabelEncoder()
data["Sex"] = le_sex.fit_transform(data["Sex"])
le_embarked = LabelEncoder()
data["Embarked"] = le_embarked.fit_transform(data["Embarked"])

# Split features and target
X = data.drop("Survived", axis=1)
y = data["Survived"]

# Optional: scale numeric features (good for non-tree models)
scaler = StandardScaler()
X[["Age", "Fare"]] = scaler.fit_transform(X[["Age", "Fare"]])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators = 50 , random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)
print("\nFeature Importances:\n", feature_importance)

# Confusion matrix and classification report
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save predictions safely
predictions = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})
predictions.to_csv("titanic_predictions.csv", index=False)
print("Predictions saved as titanic_predictions.csv in current folder.")