# Task 2 - Ticket Classification System using Machine Learning

## 📌 Project Description
This project automatically classifies customer support tickets into different categories and priority levels using Machine Learning.

The system uses Natural Language Processing (NLP) techniques to analyze text and predict:
- Ticket Category
- Ticket Priority

---

## 🎯 Objective
- Load ticket dataset
- Convert text data into numerical format
- Train classification models
- Predict ticket category and priority
- Evaluate model performance

---

## 📊 Dataset
Dataset contains customer support tickets.

### Columns:
- ticket → Support request text
- category → Ticket category (Technical, Billing, Account, Other)
- priority → Priority level (High, Medium, Low)

---

## 🛠 Libraries Used
- pandas → Data handling
- scikit-learn → Machine learning and text processing

---

## ⚙ Process / Steps
1. Load dataset using pandas
2. Convert text into numerical features using TF-IDF
3. Split data into training and testing sets
4. Train Naive Bayes classification models
5. Predict category and priority
6. Evaluate model accuracy

---

## 📈 Output
- Category prediction accuracy
- Priority prediction accuracy
- Classification report
- New ticket prediction demo

---

## ▶ How to Run

1. Install required libraries:
pip install pandas scikit-learn

2. Run program:
python tickets_classifier.py

---

## 📂 Files Included
- tickets_classifier.py → Python code
- tickets.csv → Dataset
- README.md → Project documentation

---

## 🚀 Future Improvement
- Use larger dataset for better accuracy
- Improve text preprocessing
- Try advanced NLP models
- Deploy as web application
