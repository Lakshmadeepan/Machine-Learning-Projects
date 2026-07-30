import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

data=pd.read_csv("Housing.csv")
X=data.iloc[:,1:13]
Y=data.iloc[:,0]

encoders={}
categories=["mainroad","guestroom","basement","hotwaterheating",
            "airconditioning","prefarea","furnishingstatus"
]
for column in categories:
  encoder=LabelEncoder()
  X[column]=encoder.fit_transform(X[column])
  encoders[column]=encoder



X_train,X_test,Y_train,Y_test=train_test_split(
    X,
    Y,
    test_size=0.1,
    random_state=42
)
model=LinearRegression()
model.fit(
    X_train,
    Y_train 
)
score=model.score(X_test,Y_test)
print(score*100)

joblib.dump(model,"price_prediction_model.pkl")
loaded_model=joblib.load("price_prediction_model.pkl")

sample = [[7420, 4, 2, 3, "yes", "no", "no", "no", "yes", 2, "yes", "furnished"]]
sample_data = pd.DataFrame(sample, columns=X.columns)


for column in categories:  
  sample_data[column] = encoders[column].transform(sample_data[column])


prediction = loaded_model.predict(sample_data)
print("Predicted price is:", prediction)