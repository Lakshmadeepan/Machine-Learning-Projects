import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib 

data=pd.read_csv(r"05.loan-approval_prediction\loan_approval_prediction_dataset.csv")
data.drop("Loan_ID",axis=1,inplace=True)
data["Gender"]=data["Gender"].fillna(data["Gender"].mode()[0])
data["Married"]=data["Married"].fillna(data["Married"].mode()[0])
data["Dependents"]=data["Dependents"].fillna(data["Dependents"].mode()[0])
data["Self_Employed"]=data["Self_Employed"].fillna(data["Self_Employed"].mode()[0])
data["LoanAmount"]=data["LoanAmount"].fillna(data["LoanAmount"].median())
data["Loan_Amount_Term"]=data["Loan_Amount_Term"].fillna(data["Loan_Amount_Term"].median())
data["Credit_History"]=data["Credit_History"].fillna(data["Credit_History"].median())

X=data.drop("Loan_Status",axis=1)
Y=data["Loan_Status"]

encoders={}
categories=["Gender","Married","Dependents","Education","Self_Employed","Property_Area"]
for column in categories:
    encoder=LabelEncoder()
    X[column]=encoder.fit_transform(X[column])
    encoders[column]=encoder

encoder=LabelEncoder()
Y=encoder.fit_transform(Y)

X_train,X_test,Y_train,Y_test=train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)

model=RandomForestClassifier(
    n_estimators=75,
    max_depth=7,
    max_samples=75,
    random_state=42
)

model.fit(
    X_train,
    Y_train
)

score=model.score(
    X_test,
    Y_test
)

joblib.dump(model,r"05.loan-approval_prediction\model\loan_approval_model.plk")
joblib.dump(encoders,r"05.loan-approval_prediction\model\encoders.plk")
joblib.dump(encoder,r"05.loan-approval_prediction\model\target_encoder.plk")
# print("Accuracy:",score*100)
# print(X.info())
# print(Y.info())
# print(data.info())