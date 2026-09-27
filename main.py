import pandas as pd
fake=pd.read_csv("Fake.csv")
true=pd.read_csv("True.csv")
fake["label"]=0
true["label"]=1
df=pd.concat([fake,true],ignore_index=True)
df=df.drop_duplicates()
df["content"]=df["title"]+" "+df["text"]
x=df["content"]
y=df["label"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer=TfidfVectorizer(stop_words="english",max_df=0.7)
x_train_tfidf=vectorizer.fit_transform(x_train)
x_test_tfidf=vectorizer.transform(x_test)
from sklearn.svm import LinearSVC
model=LinearSVC()
model.fit(x_train_tfidf,y_train)
y_pred=model.predict(x_test_tfidf)
from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test,y_pred)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
import joblib
joblib.dump(model,"fake_news_svm.pkl")
joblib.dump(vectorizer,"tfidf_vectorizer.pkl")
print("done")
