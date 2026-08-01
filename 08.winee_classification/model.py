import pandas as pd
import joblib 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


data=pd.read_csv(r"08.winee_classification\WineQT.csv")
data.drop("Id",axis=1,inplace=True)
X=data.drop("quality",axis=1)
Y=data["quality"]

model=KNeighborsClassifier(
    n_neighbors=5,
    weights="distance",
    metric="minkowski",
    p=2
)

X_train,X_test,Y_train,Y_test=train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)

scaler=StandardScaler()

X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

model.fit(
    X_train_scaled,
    Y_train
)
score=model.score(
    X_test_scaled,
    Y_test
)
# print(score)

joblib.dump(model,r"08.winee_classification\model\wine_classification_model.pkl")
joblib.dump(scaler,r"08.winee_classification\model\target_scaler.pkl")
