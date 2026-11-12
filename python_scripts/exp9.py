# Experiment: EXP9
# Laboratory Program for Machine Learning in Cybersecurity


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

# Sample email dataset
data = {
    "message": [
        "Win a free prize now",
        "Claim your free reward today",
        "Congratulations you won a lottery",
        "Get free money immediately",
        "Meeting scheduled for tomorrow",
        "Please submit your assignment",
        "Your project report is ready",
        "Let's discuss the meeting",
        "Free offer claim your prize",
        "Please attend the class tomorrow",
        "You won a free gift",
        "The lecture starts at ten",
        "Claim your cash reward now",
        "Please send the project file",
        "Congratulations claim your prize",
        "Your appointment is confirmed",
        "Win cash and free gifts",
        "The exam is scheduled tomorrow",
        "Get your free bonus now",
        "Please check the meeting notes"
    ],
    "label": [
        "spam", "spam", "spam", "spam", "ham",
        "ham", "ham", "ham", "spam", "ham",
        "spam", "ham", "spam", "ham", "spam",
        "ham", "spam", "ham", "spam", "ham"
    ]
}

df = pd.DataFrame(data)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    df["message"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# TF-IDF + Naive Bayes model
model = make_pipeline(
    TfidfVectorizer(),
    MultinomialNB()
)

# Train the model
model.fit(X_train, y_train)

# Test the model
predictions = model.predict(X_test)

print("Accuracy:",
      accuracy_score(y_test, predictions))

print(classification_report(
    y_test, predictions, zero_division=0
))

# Predict a new email
email = ["Congratulations! Claim your free prize now"]

result = model.predict(email)[0]

if result == "spam":
    print("Email Classification: SPAM")
else:
    print("Email Classification: NOT SPAM")




