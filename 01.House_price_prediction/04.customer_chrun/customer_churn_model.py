import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib


data=pd.read_csv(r"04.customer_chrun\customer_chrun_dataset\customer_chrun_dataset.csv")
data.drop_duplicates(inplace=True)
data.drop("customerID",axis=1,inplace=True)

data["TotalCharges"] = pd.to_numeric(
    data["TotalCharges"],
    errors="coerce"
)
data["TotalCharges"] = data["TotalCharges"].fillna(
    data["TotalCharges"].median()
)

X=data.drop("Churn",axis=1,inplace=False)
Y=data.iloc[:,-1]

encoders={}
categories=["gender","Partner","Dependents","PhoneService","MultipleLines","InternetService",
            "OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies",
            "Contract","PaperlessBilling","PaymentMethod"
            ]

for column in categories:
    encoder=LabelEncoder()
    X[column]=encoder.fit_transform(X[column])
    encoders[column]=encoder

encoder=LabelEncoder()
Y=encoder.fit_transform(Y)

joblib.dump(encoder,r"04.customer_chrun\models\target-encoder.plk")

X_train,X_test,Y_train,Y_test=train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)

model=DecisionTreeClassifier(
    max_depth=10,
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


joblib.dump(model,r"04.customer_chrun\models\customer_chrun_model.plk")
joblib.dump(encoders,r"04.customer_chrun\models\encoder.plk")


