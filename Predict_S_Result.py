from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd



data = {
    "study_hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "attendance": [40, 45, 50, 55, 60, 65, 70, 75, 80, 85],
    "previous_marks": [35, 40, 45, 50, 55, 60, 65, 70, 75, 80],
    "result": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  # 0 = Fail, 1 = Pass
}

df = pd.DataFrame(data)



print("Dataset Information :")
print(df.info(),"\n")
print("\nMissing Values :")
print(df.isnull().sum())
print("\n")


X = df[["study_hours", "attendance", "previous_marks"]]
y = df["result"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(type(X))
print(type(y))

model = LogisticRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))



new_student = pd.DataFrame({
    "study_hours": [6],
    "attendance": [80],
    "previous_marks": [70]
})



prediction = model.predict(new_student)
if prediction[0] == 1:
    print("Congratulations! Student will PASS.")
else:
    print("Unfortunately, Student will FAIL.")



sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()