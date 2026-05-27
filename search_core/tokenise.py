import string

from pathlib import Path
from nltk.stem import PorterStemmer
from search_core.constants import DATA_DIR

def get_stopwords() -> list[str]: 
    stopwords_path = DATA_DIR / "stopwords.txt"
    
    with open(stopwords_path, "r") as f:
        stopwords = f.read()
    
    stopwords = stopwords.splitlines()

    nop = []
    for word in stopwords:
        np = ""
        for char in word:
            if char not in string.punctuation:
                np += char
        nop.append(np)
    stopwords = nop

    return stopwords

def sanitise_text(text: str, stopwords: list[str]) -> list[str]:
    text = text.lower()

    np = ""
    for char in text:
        if char not in string.punctuation:
            np += char
    text = np

    tokens = text.split()
    tokens = [t for t in tokens if t not in stopwords]

    stemmer = PorterStemmer()
    return [stemmer.stem(t) for t in tokens]

def sanitise_term(term: str) -> str:
    stopwords = get_stopwords()
    sanitised_term = sanitise_text(term, stopwords)

    if len(sanitised_term) > 1:
        raise ValueError("Term results in more than a single token!")
    
    term = sanitised_term[0]

    return term