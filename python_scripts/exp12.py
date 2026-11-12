# Experiment: EXP12
# Laboratory Program for Machine Learning in Cybersecurity


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Create a synthetic WSN dataset
data = {
    "timestamp": [
        f"2026-01-{(i % 28) + 1:02d} {i % 24:02d}:00:00"
        for i in range(40)
    ],
    "source_ip": [
        "192.168.1." + str((i % 10) + 1)
        for i in range(40)
    ],
    "event_type": [
        "normal" if i % 2 == 0 else "attack"
        for i in range(40)
    ],
    "node_id": [i % 10 + 1 for i in range(40)],
    "rssi": [
        -50 if i % 2 == 0 else -90
        for i in range(40)
    ],
    "packet_size": [
        100 if i % 2 == 0 else 1500
        for i in range(40)
    ],
    "label": [
        0 if i % 2 == 0 else 1
        for i in range(40)
    ]
}

df = pd.DataFrame(data)

# Save dataset as CSV
df.to_csv("wsn_data.csv", index=False)

print("Dataset created successfully!")
print(df.head())

# Step 2: Extract time-based features
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["hour"] = df["timestamp"].dt.hour
df["day"] = df["timestamp"].dt.dayofweek

# Step 3: Encode categorical features
df["source_ip"] = df["source_ip"].astype("category").cat.codes
df["event_type"] = df["event_type"].astype("category").cat.codes

# Step 4: Select features and target
features = [
    "hour", "day", "source_ip",
    "event_type", "node_id", "rssi", "packet_size"
]

X = df[features]
y = df["label"]

# Step 5: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Step 6: Train Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Step 7: Evaluate model
predictions = model.predict(X_test)

print("\nAccuracy:",
      accuracy_score(y_test, predictions))

print("\nClassification Report:")
print(classification_report(
    y_test, predictions,
    zero_division=0
))

# Step 8: Predict a new network event
new_event = X_test.iloc[[0]]

result = model.predict(new_event)[0]

if result == 1:
    print("Alert: Malicious Event Detected")
else:
    print("Normal Network Event")




