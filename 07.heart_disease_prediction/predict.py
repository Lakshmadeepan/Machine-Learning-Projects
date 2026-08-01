import pandas as pd
import joblib

loaded_model=joblib.load(r"07.heart_disease_prediction\models\hear_disease_prediction_model.plk")
scalar=joblib.load(r"07.heart_disease_prediction\models\scalar.plk")
sample = [[
    67,   # age
    1,    # sex (Male)
    3,    # cp
    160,  # trestbps
    286,  # chol
    1,    # fbs
    0,    # restecg
    108,  # thalachh
    1,    # exang
    3.5,  # oldpeak
    0,    # slope
    3,    # ca
    3     # thal
]]
columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalachh',
       'exang', 'oldpeak', 'slope', 'ca', 'thal']
sample_data=pd.DataFrame(sample,columns=columns)
sample_scalar=scalar.transform(sample_data)
prediction=loaded_model.predict(sample_scalar)
# print(prediction)
# print(loaded_model.predict_proba(sample_scalar))
if prediction[0]==0:
    print("The heart  disease risk is high")
else:
    print("The heart  disease risk is low")
