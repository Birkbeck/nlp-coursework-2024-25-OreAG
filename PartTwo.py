import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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
print("Random Forest Classification Report:\n", classification_report(y_test, rf_preds))
