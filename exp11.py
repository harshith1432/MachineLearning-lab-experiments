# Experiment: EXP11
# Laboratory Program for Machine Learning in Cybersecurity


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Synthetic database activity dataset
data = {
    "query_length": [
        25, 30, 40, 35, 50, 45, 20, 60, 28, 38,
        95, 120, 110, 100, 130, 90, 115, 105, 125, 98
    ],
    "special_chars": [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    ],
    "suspicious_keyword": [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    ],
    "failed_attempts": [
        0, 1, 0, 1, 0, 2, 0, 1, 0, 2,
        3, 5, 4, 3, 6, 4, 5, 3, 6, 4
    ],
    "label": [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    ]
}

df = pd.DataFrame(data)

# Separate features and labels
X = df.drop("label", axis=1)
y = df["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3,
    random_state=42, stratify=y
)

# Create and train Decision Tree model
model = DecisionTreeClassifier(
    max_depth=3, random_state=42
)

model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)

print("Accuracy:",
      accuracy_score(y_test, predictions))

print(classification_report(
    y_test, predictions, zero_division=0
))

# Test a new database event
event = pd.DataFrame([{
    "query_length": 110,
    "special_chars": 1,
    "suspicious_keyword": 1,
    "failed_attempts": 4
}])

result = model.predict(event)[0]

if result == 1:
    print("Alert: Malicious Data Injection Detected")
else:
    print("Database Activity: Normal")




