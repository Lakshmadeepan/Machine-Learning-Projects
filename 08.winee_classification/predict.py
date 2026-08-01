import pandas as pd
import joblib

loaded_model=joblib.load(r"08.winee_classification\model\wine_classification_model.plk")
scaler=joblib.load(r"08.winee_classification\model\target_scaler.plk")
columns=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol']

sample= [[9.2, 0.43, 0.52, 2.3, 0.083, 14.0, 23.0, 0.9976, 3.35, 0.61, 11.3]]



sample_data=pd.DataFrame(sample,columns=columns)
sample_data=scaler.transform(sample_data)
prediction=loaded_model.predict(sample_data)
print("prediction:",prediction[0])