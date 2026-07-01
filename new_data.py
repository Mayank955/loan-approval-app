import pandas as pd
import joblib
loaded_model = joblib.load("loan_model.pkl")

new_applicant = pd.DataFrame({
    "income": [60000],
    "credit_score": [720],
    "loan_amount": [250000]
})

prediction = loaded_model.predict(new_applicant)
if prediction[0] == 1:
    print("\nLoan Approved")
else:
    print("\nLoan Rejected")
