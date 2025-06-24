import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, classification_report


df = pd.read_csv("p2_texts/hansard40000.csv")

#renaming party name to Labour
df['party'] = df['party'].replace({'Labour (Co-op)' : 'Labour'})

#remove rows excl top 4 most common parties and speaker value
top_parties = df['party'].value_counts().drop('Speaker').nlargest(4).index.tolist()
df = df[df['party'].isin(top_parties)]

#remove rows where 'speech_class' not speech
df = df[df['speech_class'] == 'Speech']

#remove rows where text in speech less than 1000
df = df[df['speech'].str.len() >= 1000]

print(df.shape)


#Vectorising the speeches
vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
X = vectorizer.fit_transform(df['speech'])
y = df['party']

#splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=26, stratify=y)

#training random forest
rf = RandomForestClassifier(n_estimators=300,random_state=26)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
print("Random Forest F1 score:", f1_score(y_test, rf_preds, average='macro'))
print("Random Forest Classification Report:\n", classification_report(y_test, rf_preds, zero_division=0))

#SVM
svm = SVC(kernel='linear', random_state=26)
svm.fit(X_train, y_train)
svm_preds = svm.predict(X_test)
print("SVM F1 Score:", f1_score(y_test, svm_preds, average='macro'))
print("SVM Classification Report:\n", classification_report(y_test, svm_preds))

#ngram

vectorizer_ngram =TfidfVectorizer(stop_words='english', max_features=3000, ngram_range=1,3)
X_ngram = vectorizer_ngram.fit_transform(df['speech'])

X_train_ng, X_test_ng, y_train_ng, y_test_ng = train_test_split(X_ngram, y, test_size=0.2, random_state=26,stratify=y)

#train and print again
rf_ng = RandomForestClassifier(n_estimators=300, random_state=26)
rf_ng.fit(X_train_ng, y_train_ng)
rf_ng_preds = rf_ng.predict(X_test_ng)

print("Random Forest(n_gram Classification Report:\n", classification_report(y_test_ng, rf_ng_preds))
