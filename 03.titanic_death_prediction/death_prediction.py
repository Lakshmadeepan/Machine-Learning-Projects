import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

data=pd.read_csv(r"03.titanic_death_prediction\Titanic-Dataset.csv")

data["Age"]=data["Age"].fillna(data["Age"].median())
data["Fare"]=data["Fare"].fillna(data["Fare"].median())
data["Embarked"]=data["Embarked"].fillna(data["Embarked"].mode()[0])
data.drop("Name",axis=1,inplace=True)
data.drop("Cabin",axis=1,inplace=True)
data.drop("PassengerId",axis=1,inplace=True)
data.drop("Ticket",axis=1,inplace=True)
data.drop_duplicates(inplace=True)
# print(data.info())
# print(data.isnull().sum())

X=data.iloc[:,1:]
Y=data.iloc[:,0]
# print(X.info())
encoders={}

categories=["Sex","Embarked"]
for column in categories:
  encoder=LabelEncoder()
  X[column]=encoder.fit_transform(X[column])
  encoders[column]=encoder


X_train,X_test,Y_train,Y_test=train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)

model=LogisticRegression(max_iter=500)
model.fit(
    X_train,
    Y_train
)
score=model.score(
    X_test,
    Y_test
)

joblib.dump(model,"titanic_death_prediction.pkl")
loaded_model=joblib.load("titanic_death_prediction.pkl")

print("Accuracy:",score*100)
# print(data.info())
sample = [[1, "male", 45, 1, 0, 90.0, "C"]]

sample_data=pd.DataFrame(sample,columns=X.columns)
for column in categories: 
  sample_data[column]=encoders[column].transform(sample_data[column])
prediction=loaded_model.predict(sample_data)
print(f"The predicted output is {prediction}")