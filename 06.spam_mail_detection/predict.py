import pandas as pd
import joblib

loaded_model=joblib.load(r"06.spam_mail_detection\model.plk\mail_span_detection_model.plk")
vectorizer=joblib.load(r"06.spam_mail_detection\model.plk\vectorizer.plk")
encoder=joblib.load(r"06.spam_mail_detection\model.plk\target_encoder.plk")

data=input("Enter the message:\n")
sample=[data]
sample_data=vectorizer.transform(sample)
prediction=loaded_model.predict(sample_data)
predicted=encoder.inverse_transform(prediction)
if predicted[0] == "spam":
    print("🚨 Spam Message")
else:
    print("✅ Ham (Safe Message)")