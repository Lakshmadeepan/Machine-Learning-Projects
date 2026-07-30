import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load model
model = joblib.load(r"02.Ires_flower_classification\models\ires_model_prediction.pkl")

# Load encoder
encoder = joblib.load(r"02.Ires_flower_classification\models\encoders.pkl")

# Sample flower
sample = [[5.7, 2.8, 4.1, 1.3]]

columns = [
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm"
]

sample_data = pd.DataFrame(sample, columns=columns)

prediction = model.predict(sample_data)

flower = encoder.inverse_transform(prediction)

print("Predicted Flower:", flower[0])