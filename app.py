import joblib
import streamlit as s

model = joblib.load("loan_model.pkl")
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

