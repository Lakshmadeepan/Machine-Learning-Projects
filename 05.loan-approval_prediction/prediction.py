import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

loaded_model=joblib.load(r"05.loan-approval_prediction\model\loan_approval_model.plk")
encoders=joblib.load(r"05.loan-approval_prediction\model\encoders.plk")
encoder=joblib.load(r"05.loan-approval_prediction\model\target_encoder.plk")
columns=['Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term', 'Credit_History', 'Property_Area']

categories=["Gender","Married","Dependents","Education","Self_Employed","Property_Area"]
sample = [[
    "Male",          # Gender
    "Yes",           # Married
    "1",               # Dependents
    "Graduate",      # Education
    "No",            # Self_Employed
    5000,            # ApplicantIncome
    2000,            # CoapplicantIncome
    150,             # LoanAmount
    360,             # Loan_Amount_Term
    1.0,             # Credit_History
    "Urban"          # Property_Area
]]
sample_data=pd.DataFrame(sample,columns=columns)
for column in categories:
    sample_data[column]=encoders[column].transform(sample_data[column])

prediction=loaded_model.predict(sample_data)
predicted=encoder.inverse_transform(prediction)
if predicted[0]=="Y":
    print("Loan status:approved")

else:
    print("Loan status:rejected")