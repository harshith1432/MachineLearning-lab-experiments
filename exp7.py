# Experiment: EXP7
# Laboratory Program for Machine Learning in Cybersecurity


from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Load the built-in digit image dataset
digits = load_digits()
X = digits.data
y = digits.target

# Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42,
    stratify=y
)

# Train the machine learning model
model = RandomForestClassifier(
    n_estimators=100, random_state=42
)
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Test accuracy: {accuracy:.2%}")

# Demonstrate character recognition
sample = X_test[:5]
actual = y_test[:5]
predicted = model.predict(sample)

for i in range(len(sample)):
    print(
        f"Actual: {actual[i]}, "
        f"Predicted: {predicted[i]}"
    )




