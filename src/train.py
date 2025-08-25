import pandas as pd # pyright: ignore[reportMissingModuleSource]
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
from sklearn.linear_model import LogisticRegression # type: ignore
from sklearn.pipeline import Pipeline # type: ignore
import joblib # type: ignore

#import dataset
df = pd.read_csv('../data/transactions_sample.csv')

X = df["transaction"]
y = df["category"]

#train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#build pipeline (TF-IDF + LOGISTIC REGRESSION)
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression(max_iter=1000))
])

#train model
model.fit(X_train, y_train)

#evaluate
print("training accuracy:", model.score(X_train, y_train))
print("test accuracy:", model.score(X_test, y_test))
 #save model
joblib.dump(model, "../models/model.joblib")
print("model saved to ../models/model.joblib")