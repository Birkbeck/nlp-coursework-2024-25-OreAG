import pandas as pd

df = pd.read_csv("p2_texts/hansard40000.csv")

#renaming party name to Labour
df['party'] = df['party'].replace({'Labour (Co-op)' : 'Labour'})

#remove rows excl top 4 most common parties and speaker value
top_parties = df['party'].value_counts().drop('Speaker').nlargest(4).index.tolist()
df = df[df['party'].isin(top_parties)]

#remove rows where 'speech_class' not speech
