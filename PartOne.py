#Re-assessment template 2025

# Note: The template functions here and the dataframe format for structuring your solution is a suggested but not mandatory approach. You can use a different approach if you like, as long as you clearly answer the questions and communicate your answers clearly.

import pandas as pd
from pathlib import Path
import nltk
from nltk.corpus import cmudict
from nltk.tokenize import word_tokenize, sent_tokenize
import string
import spacy, pickle
import math
from collections import Counter


nltk.download('punkt')
nltk.download('cmudict')

cmu_dict = cmudict.dict()


nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000

def read_novels(novels_dir= "p1_texts/novels"):
    novels_path = Path(novels_dir)
    all_files = list(novels_path.glob('*.txt'))
    records = []

    if not all_files:
        print("No txt files found")
        return pd.DataFrame()

    for file in all_files:
        try:
            parts = file.stem.replace("_"," ").split('-')
            if len(parts) <3:
                continue #skip files that don't match format
            year = int(parts[-1])
            author = parts[-2]
            title = '-'.join(parts[:-2])
            text = file.read_text(encoding= 'utf-8')
            records.append({'text': text, 'title': title, 'author': author, 'year': year})
        except Exception as e:
            print(f"Error processing {file.name}: {e}")
            continue #skip files with issues

    df = pd.DataFrame(records)
    if 'year' in df.columns:
        df = df.sort_values(by= 'year').reset_index(drop=True)

    return df
def count_syl(word, d):
    word = word.lower()
    if word in d:
        #Return min syllable count across all pronunciations
        return min([len([syl  for syl in pron if syl[-1].isdigit()]) for pron in d[word]])
    return 1


def fk_level(text, d):
    sentences = sent_tokenize(text)
    words =[w.lower() for w in word_tokenize (text) if w.isalpha()]
    num_sentences = len(sentences)
    num_words = len(words)
    num_syllables = sum(count_syl(w,d) for w in words)

    if num_sentences == 0 or num_words ==0:
        return 0

    score = 0.39 * (num_words /num_sentences) + 11.8 * (num_syllables/ num_words) - 15.59
    return round(score, 2)

def nltk_ttr(text):
        #tokenize
        tokens = word_tokenize(text)
         #take away the punctuations and make texts lowercase
        words= [token.lower() for token in tokens if token.isalpha()]
        types = set(words)
        return len(types) /len(words) if words else 0


def get_ttrs(df):
    return {row["title"]: nltk_ttr(row["text"]) for _, row in df.iterrows()}


def get_fks(df):
    return {row["title"]: fk_level(row["text"]) for _, row in df.iterrows()}


def parse(df, store_path=Path.cwd() / "pickles", out_name="parsed.pickle"):
    parsed_docs = []

    for text in df['text']:
        if len(text) > nlp.max_length:
            print("Warning: text too long")
            text = text[:nlp.max_length]
        parsed_docs.append(nlp(text))

    df['parsed'] = parsed_docs

    store_path.mkdir(parents=True, exist_ok=True)
    with open(store_path/out_name, 'wb') as f:
        pickle.dump(df, f)

    return df

def subjects_by_verb_count(doc, verb):
    subjects = []
    for token in doc:
        if token.lemma_ == verb and token.pos_ == "VERB":
            for child in token.children:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    subjects.append(child.text.lower())
    return Counter(subjects).most_common(10)

def subjects_by_verb_pmi(doc, target_verb):
    total_tokens = len([token for token in doc if token.is_alpha])
    subj_counter = Counter()
    subj_joint_counter = Counter()
    verb_counter = 0

    for token in doc:
        if token.lemma_ == target_verb and token.pos_ == "VERB":
            verb_counter += 1
            for child in token.children:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    subj = child.text.lower()
                    subj_counter[subj] +=1
                    subj_joint_counter[subj] +=1
    pmi_scores = {}
    for subj, joint_count in subj_joint_counter.items():
        p_subj = subj_counter[subj] / total_tokens
        p_verb = verb_counter /total_tokens
        p_joint = joint_count / total_tokens
        if p_subj * p_verb > 0:
            pmi = math.log2(p_joint/ (p_subj * p_verb))
            pmi_scores[subj] = pmi

    sorted_pmi = sorted(pmi_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_pmi[:10]


def adjective_counts(doc):
    adjectives = [token.text.lower() for token in doc if token.pos_ == "ADJ"]
    return Counter(adjectives).most_common(10)

def object_counts(doc):
    objects = [token.text.lower() for token in doc if token.dep_ =="dobj"]
    return Counter(objects).most_common(10)


if __name__ == "__main__":
    path = Path.cwd() / "p1_texts" / "novels"
    print(path)
    df = read_novels(path) # this line will fail until you have completed the read_novels function above.
    print(df.head())
    parse(df)
    print(df.head())
    print(get_ttrs(df))
    print(get_fks(df))
    df = pd.read_pickle(Path.cwd() / "pickles" /"name.pickle")
    print(adjective_counts(df.loc[0, "parsed"]))

    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_count(row["parsed"], "hear"))
        print("\n")

    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_pmi(row["parsed"], "hear"))
        print("\n")


