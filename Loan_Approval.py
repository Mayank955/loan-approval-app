import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import streamlit as s


data = {
    "income": [50000, 25000, 80000, 40000, 90000, 30000, 70000, 35000, 95000, 45000],
    "credit_score": [750, 550, 800, 650, 820, 580, 720, 600, 850, 670],
    "loan_amount": [200000, 500000, 300000, 250000, 400000, 450000, 280000, 350000, 320000, 270000],
    "loan_status": [1, 0, 1, 1, 1, 0, 1, 0, 1, 1]  
}


df = pd.DataFrame(data)
print("Dataset Information :")
print(df.info())
print("\n")
print("Missing Values :")
print(df.isnull().sum())
print("\n")



X = df[["income", "credit_score", "loan_amount"]]
y = df["loan_status"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


model = LogisticRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)


print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(cm)
print("\n")


new_applicant = pd.DataFrame({
    "income": [60000],
    "credit_score": [720],
    "loan_amount": [250000]
})


prediction = model.predict(new_applicant)

if prediction[0] == 1:
    print("Loan Approved.")
else:
    print("Loan Rejected.")



sns.heatmap(cm, annot=True, cmap="Blues")
plt.title("Loan Approval Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()




s.title(":blue[Loan Approval Predictor:]")

income = s.slider("Income : ", 20000, 100000)
credit_score = s.slider("Credit Score : ", 300, 850)
loan_amount = s.slider("Loan Amount : ", 10000, 500000)

prediction = model.predict([[income, credit_score, loan_amount]])

# Success
if prediction[0] == 1:
    s.success("Loan Approved")
else:
    s.error("Loan Rejected")

# Write
if prediction[0] == 1:
    s.write("Prediction : Pass")
else:
    s.write("Prediction : Fail")


s.title("_Streamlit_ is :blue[cool] :sunglasses:")

joblib.dump(model, "loan_model.pkl")
print("\nModel saved successfully as loan_model.pkl")

# cd "C:\Users\nemsi\OneDrive\Desktop\Python_Learn\JASIQ_LABS_PY\Phase 4\Capstone Projects"
# streamlit run app.py