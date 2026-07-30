from numpy.random.mtrand import sample
import pandas as pd 
import numpy as np
from sklearn.model_selection import  train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib

data=pd.read_csv(r"02.Ires_flower_classification\Iris.csv")
X=data.iloc[:,1:5]#column
Y=data.iloc[:,-1]
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

joblib.dump(model,r"02.Ires_flower_classification\models\ires_model_prediction.pkl")
joblib.dump(encoder,r"02.Ires_flower_classification\models\encoders.pkl")
