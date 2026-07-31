import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

data=pd.read_csv(r"06.spam_mail_detection\spam_datset\spam_ham_dataset.csv")
data.drop("dummy",axis=1,inplace=True)
data.drop("label_num",axis=1,inplace=True)
X=data["text"]
Y=data["label"]
encoder=LabelEncoder()
Y=encoder.fit_transform(Y)
vectorizer=CountVectorizer()
X=vectorizer.fit_transform(X)
X_train,X_test,Y_train,Y_test=train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)
model=MultinomialNB()
model.fit(
    X_train,
    Y_train
)
score=model.score(
    X_test,
    Y_test
)

joblib.dump(model,r"06.spam_mail_detection\model.plk\mail_span_detection_model.plk")
joblib.dump(vectorizer,r"06.spam_mail_detection\model.plk\vectorizer.plk")
joblib.dump(encoder,r"06.spam_mail_detection\model.plk\target_encoder.plk")

# print("Score:",score*100)
# print(data.info())