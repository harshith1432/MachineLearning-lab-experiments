# Experiment: EXP6
# Laboratory Program for Machine Learning in Cybersecurity

import pandas as pd
from sklearn.ensemble import IsolationForest

# Load pre-built dataset
df = pd.read_csv("PhiUSIIL_Phishing_URL_Dataset.csv")

# Feature engineering
df["DigitRatio"] = df["NoOfDegitsInURL"] / (df["NoOfLettersInURL"] + 1)
df["SpecialRatio"] = df["NoOfOtherSpecialCharsInURL"] / (df["URLLength"] + 1)

# Select features
features = [
    "URLLength", "DomainLength", "NoOfSubDomain",
    "NoOfDegitsInURL", "NoOfOtherSpecialCharsInURL",
    "SpacialCharRatioInURL", "IsDomainIP", "IsHTTPS",
    "HasObfuscation", "NoOfURLRedirect",
    "DigitRatio", "SpecialRatio"
]

X = df[features].fillna(0)

# Isolation Forest
model = IsolationForest(contamination=0.05, random_state=42)
df["Anomaly"] = model.fit_predict(X)

# Output
print("Total URLs:", len(df))
print("Normal URLs:", (df["Anomaly"] == 1).sum())
print("Abnormal URLs:", (df["Anomaly"] == -1).sum())

print("\nAbnormal URL Samples:")
print(df[df["Anomaly"] == -1][["URL", "Anomaly"]].head(10))

import matplotlib.pyplot as plt

plt.scatter(
    df["URLLength"],
    df["NoOfSubDomain"],
    c=df["Anomaly"]
)

plt.xlabel("URL Length")
plt.ylabel("Number of Subdomains")
plt.title("Abnormal URL Pattern Detection")
plt.show()



