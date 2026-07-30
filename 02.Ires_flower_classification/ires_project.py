from numpy.random.mtrand import sample
import pandas as pd 
import numpy as np
from sklearn.model_selection import  train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib

data=pd.read_csv("Iris.csv")
X=data.iloc[:,1:5].values
Y=data.iloc[:,-1].values
encoder=LabelEncoder()
Y=encoder.fit_transform(Y)
X_train,X_test,Y_train,Y_test=train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)
model=LogisticRegression()
model.fit(
    X_train,
    Y_train
)
score=model.score(
    X_test,
    Y_test
)

joblib.dump(model,"ires_model_prediction.pkl")
loaded_model=joblib.load("ires_model_prediction.pkl")

print("Score:",score*100)



sample=[[5.7,2.8,4.1,1.3]]
sample_data=pd.DataFrame(sample)
prediction=loaded_model.predict(sample_data)
flower=encoder.inverse_transform(prediction)
print(flower)
