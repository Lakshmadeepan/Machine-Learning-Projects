import pandas as pd
import joblib

# Load model
model = joblib.load(r"03.titanic_death_prediction\models\titanic_death_prediction.pkl")

# Load encoders
encoders = joblib.load(r"03.titanic_death_prediction\models\encoders.pkl")

# Sample passenger
sample = [[1, "female", 25, 0, 0, 80.0, "C"]]

columns = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked"
]

sample_data = pd.DataFrame(sample, columns=columns)

# Encode categorical columns
categories = ["Sex", "Embarked"]

for column in categories:
    sample_data[column] = encoders[column].transform(sample_data[column])

# Predict
prediction = model.predict(sample_data)

if prediction[0] == 1:
    print("Passenger Survived")
else:
    print("Passenger Did Not Survive")