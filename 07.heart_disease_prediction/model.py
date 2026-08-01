import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

# from sklearn.linear_model import LogisticRegression  Score2: 71.95767195767195
# from sklearn.ensemble import RandomForestClassifier low accuracy tested Score2: 68.51851851851852
# from sklearn.naive_bayes import MultinomialNB Score2: 65.34391534391534

data=pd.read_csv(r"07.heart_disease_prediction\cleaned_merged_heart_dataset.csv")
X=data.drop("target",axis=1)
Y=data["target"]
scalar=StandardScaler()
X_train,X_test,Y_train,Y_test=train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)
X_train=scalar.fit_transform(X_train)
X_test=scalar.transform(X_test)
model=SVC(
    C=1,
    kernel="rbf",
    probability=True
)
model.fit(
    X_train,
    Y_train
)

score=model.score(
    X_test,
    Y_test
)

joblib.dump(model,r"07.heart_disease_prediction\models\hear_disease_prediction_model.plk")
joblib.dump(scalar,r"07.heart_disease_prediction\models\scalar.plk")

# pred = model.predict(X_test)

# print(confusion_matrix(Y_test, pred))

print("Model loaded Succesfully")
# print(Y.value_counts())
# print(X.columns)
# print(X.info())
# # print(data.info())