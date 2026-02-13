import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load data
data = pd.read_csv("tickets.csv")

X = data["ticket"]

y_category = data["category"]
y_priority = data["priority"]

# Vectorization
vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,3))
X_vec = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_cat_train, y_cat_test = train_test_split(
    X_vec, y_category, test_size=0.25, random_state=42
)

_, _, y_pri_train, y_pri_test = train_test_split(
    X_vec, y_priority, test_size=0.25, random_state=42
)

# Models
category_model = MultinomialNB()
priority_model = MultinomialNB()

category_model.fit(X_train, y_cat_train)
priority_model.fit(X_train, y_pri_train)

# Predictions
cat_pred = category_model.predict(X_test)
pri_pred = priority_model.predict(X_test)

# Evaluation
print("CATEGORY MODEL ACCURACY:", accuracy_score(y_cat_test, cat_pred))
print(classification_report(y_cat_test, cat_pred))

print("PRIORITY MODEL ACCURACY:", accuracy_score(y_pri_test, pri_pred))
print(classification_report(y_pri_test, pri_pred))

# Demo Prediction
new_ticket = ["Payment deducted but invoice not generated"]
new_vec = vectorizer.transform(new_ticket)

print("\nNew Ticket:", new_ticket[0])
print("Predicted Category:", category_model.predict(new_vec)[0])
print("Predicted Priority:", priority_model.predict(new_vec)[0])
