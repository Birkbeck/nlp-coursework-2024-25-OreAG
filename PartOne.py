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

def read_novels(novels_dir= "texts/novels"):
    """Reads texts from a directory of .txt files and returns a DataFrame with the text, title,
    author, and year"""
    records = []
    novels_path = Path(novels_dir)

    for file in novels_path.glob('*.txt'):
        try:
            parts = file.stem.split('-')
            year = int(parts[-1])
            author = parts[-2]
            title = '-'.join(parts[:-2])
            text = file.read_text(encoding= 'utf-8')

            records.append({'text': text, 'title': title, 'author': author, 'year': year})

        except Exception as e:
            print(f"Error processing {file.name}: {e}")
    df = pd.DataFrame(records)
    df = df.sort_values(by='year').reset_index(drop=True)
    return df

def fk_level(text, d):
    """Returns the Flesch-Kincaid Grade Level of a text (higher grade is more difficult).
    Requires a dictionary of syllables per word.

    Args:
        text (str): The text to analyze.
        d (dict): A dictionary of syllables per word.

    Returns:
        float: The Flesch-Kincaid Grade Level of the text. (higher grade is more difficult)
    """
    pass


def count_syl(word, d):
    """Counts the number of syllables in a word given a dictionary of syllables per word.
    if the word is not in the dictionary, syllables are estimated by counting vowel clusters

    Args:
        word (str): The word to count syllables for.
        d (dict): A dictionary of syllables per word.

    Returns:
        int: The number of syllables in the word.
    """
    word = word.lower()
    if word in cmu_dict:
        #Return min syllable count across all pronunciations
        return min([len([syl  for syl in pron if syl[-1].isdigit()]) for pron in cmu_dict[word]])

    else:
        return 1
    pass
def flesch_kincaid(df):
    fk_scores = {}

    for _, row in df.iterrows():
        title = row['title']
        text = row['text']

        #tokenize into sentences and words
        sentences = sent_tokenize(text)
        words = [word.lower() for word in word_tokenize(text) if word.isalpha()]

        #Basic counts
        num_sentences = len(sentences)
        num_words = len(words)
        num_syllables = sum(count_syl(word) for word in words)

        if num_sentences == 0 or num_words == 0:
            fk_scores[title]= 0
            continue
        #Flesch-kincaid grade level formula
        score = 0.39 * (num_words/ num_sentences) + 11.8 * (num_syllables/ num_words) - 15.59
        fk_scores[title] = round(score,2)
    return fk_scores





    pass


def parse(df, store_path=Path.cwd() / "pickles", out_name="parsed.pickle"):
    """Parses the text of a DataFrame using spaCy, stores the parsed docs as a column and writes 
    the resulting  DataFrame to a pickle file"""
    parsed_docs = []

    for text in df['text']:
        if len(text) > nlp.max_length:
            print("Warning: text too long")
            text = text[:nlp.max_length]
            doc = nlp(text)
            parsed_docs.append(doc)
    df['parsed'] = parsed_docs

    pickle_path = store_path/ out_name
    with open(pickle_path, 'wb') as f:
        pickle.dump(df, f)
    return df
    pass


def nltk_ttr(text):
    """Calculates the type-token ratio of a text. Text is tokenized using nltk.word_tokenize."""
    ttr_dict = {}
    for _, row in df.iterrows():
        title = row['title']
        text = row['text']

        #tokenize
        tokens = word_tokenize(text)

        #take away the punctuations and make texts lowercase
        words= [token.lower() for token in tokens if token.isalpha()]

        #Calculate TTR
        types = set(words)
        ttr = len(types) /len(words) if words else 0

        ttr_dict[title] = ttr

    return ttr_dict
    pass

def get_ttrs(df):
    """helper function to add ttr to a dataframe"""
    results = {}
    for i, row in df.iterrows():
        results[row["title"]] = nltk_ttr(row["text"])
    return results


def get_fks(df):
    """helper function to add fk scores to a dataframe"""
    results = {}
    cmudict = nltk.corpus.cmudict.dict()
    for i, row in df.iterrows():
        results[row["title"]] = round(fk_level(row["text"], cmudict), 4)
    return results


def subjects_by_verb_pmi(doc, target_verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    total_tokens = len([token for token in doc if token.is_aplha])
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

    pass



def subjects_by_verb_count(doc, verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    subjects = []
    for token in doc:
        if token.lemma_ == verb and token.pos_ == "VERB":
            for child in token.children:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    subjects.append(child.text.lower())
    return Counter(subjects).most_common(10)
    pass



def adjective_counts(doc):
    """Extracts the most common adjectives in a parsed document. Returns a list of tuples."""
    adjectives = [token.text.lower() for token in doc if token.pos_ == "ADJ"]
    return Counter(adjectives).most_common(10)
    pass



if __name__ == "__main__":
    """
    uncomment the following lines to run the functions once you have completed them
    """
    #path = Path.cwd() / "p1-texts" / "novels"
    #print(path)
    #df = read_novels(path) # this line will fail until you have completed the read_novels function above.
    #print(df.head())
    #nltk.download("cmudict")
    #parse(df)
    #print(df.head())
    #print(get_ttrs(df))
    #print(get_fks(df))
    #df = pd.read_pickle(Path.cwd() / "pickles" /"name.pickle")
    # print(adjective_counts(df))
    """ 
    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_count(row["parsed"], "hear"))
        print("\n")

    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_pmi(row["parsed"], "hear"))
        print("\n")
    """

