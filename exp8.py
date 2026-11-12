# Experiment: EXP8
# Laboratory Program for Machine Learning in Cybersecurity


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Synthetic demonstration dataset
data = {
    "sender_match": [1, 1, 0, 0, 1, 0, 1, 0, 1, 0,
                     1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    "spf_pass":     [1, 1, 0, 0, 1, 0, 1, 0, 1, 0,
                     1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    "dkim_pass":    [1, 1, 0, 0, 1, 1, 1, 0, 1, 0,
                     1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    "dmarc_pass":   [1, 1, 0, 0, 1, 0, 1, 0, 1, 0,
                     1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    "reply_mismatch": [0, 0, 1, 1, 0, 1, 0, 1, 0, 1,
                       0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    "label":        [0, 0, 1, 1, 0, 1, 0, 1, 0, 1,
                     0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

X = df.drop("label", axis=1)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions, zero_division=0))

# Test one new illustrative email record
email = pd.DataFrame([{
    "sender_match": 0,
    "spf_pass": 0,
    "dkim_pass": 0,
    "dmarc_pass": 0,
    "reply_mismatch": 1
}])

result = model.predict(email)[0]

if result == 1:
    print("Alert: Suspicious spoofing pattern detected")
else:
    print("No spoofing pattern detected")





