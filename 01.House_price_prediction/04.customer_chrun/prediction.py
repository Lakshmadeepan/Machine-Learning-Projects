import joblib
from sklearn.preprocessing import LabelEncoder
import pandas as pd

loaded_model=joblib.load(r"04.customer_chrun\models\customer_chrun_model.plk")
encoders=joblib.load(r"04.customer_chrun\models\encoder.plk")
encoder=joblib.load(r"04.customer_chrun\models\target-encoder.plk")

columns=['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
       'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
       'MonthlyCharges', 'TotalCharges']
sample = [[
    "Male",
    1,
    "No",
    "No",
    1,
    "Yes",
    "No",
    "Fiber optic",
    "No",
    "No",
    "No",
    "No",
    "Yes",
    "Yes",
    "Month-to-month",
    "Yes",
    "Electronic check",
    99.90,
    99.90
]]
categories=["gender","Partner","Dependents","PhoneService","MultipleLines","InternetService",
            "OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies",
            "Contract","PaperlessBilling","PaymentMethod"
            ]

sample_data=pd.DataFrame(sample,columns=columns)
for column in categories:
    sample_data[column]=encoders[column].transform(sample_data[column])

prediction=loaded_model.predict(sample_data)
predicted=encoder.inverse_transform(prediction)
print("Prediction:",predicted)
