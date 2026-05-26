import pickle
import json
import math

from pathlib import Path
from collections import Counter
from tqdm import tqdm

from tokenise import sanitise_text, get_stopwords, sanitise_term
from constants import BM25_K1

class InvertedIndex:
    def __init__(self):
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, dict] = {}
        self.term_frequencies: dict[int, Counter] = {}
        self.stopwords = get_stopwords()
        self.doc_lengths: dict[int, int] = {}

    def __add_document(self, doc_id: int, text: str):
        tokenised_text = sanitise_text(text, self.stopwords)

        self.doc_lengths[doc_id] = len(tokenised_text)

        if doc_id not in self.term_frequencies.keys():
            self.term_frequencies[doc_id] = Counter()

        for token in tokenised_text:
            if token in self.index.keys():
                self.index[token].add(doc_id)
            else:
                self.index[token] = {doc_id}
            if token in self.term_frequencies[doc_id].keys():
                self.term_frequencies[doc_id][token] += 1
            else:
                self.term_frequencies[doc_id][token] = 1

    def get_documents(self, term: str) -> list[int]:
        term = term.lower()
        
        if term in self.index.keys():
            doc_set = self.index[term]
            doc_list = list(doc_set)

            return sorted(doc_list)
        else:
            raise ValueError("term not in index!")

    def build(self):
        movie_path = Path("data/movies.json")

        with open(movie_path, "r") as f:
            movie_data = json.load(f)

        for movie in tqdm(movie_data["movies"]):
            doc_id = movie["id"]
            self.docmap[doc_id] = movie
            self.__add_document(doc_id, f"{movie['title']} {movie['description']}")

    def save(self):
        index_cache_path = Path("cache/index.pkl")
        docmap_cache_path = Path("cache/docmap.pkl")
        term_freq_cache_path = Path("cache/term_frequencies.pkl")
        doc_lengths_path = Path("cache/doc_lengths.pkl")

        index_cache_path.parent.mkdir(parents=True, exist_ok=True)

        with open(index_cache_path, "wb") as f:
            pickle.dump(self.index, f)
        
        with open(docmap_cache_path, "wb") as f:
            pickle.dump(self.docmap, f)

        with open(term_freq_cache_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)

        with open(doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self):
        index_cache_path = Path("cache/index.pkl")
        docmap_cache_path = Path("cache/docmap.pkl")
        term_freq_cache_path = Path("cache/term_frequencies.pkl")
        doc_lengths_path = Path("cache/doc_lengths.pkl")

        try:
            with open(index_cache_path, "rb") as f:
                self.index = pickle.load(f)
            
            with open(docmap_cache_path, "rb") as f:
                self.docmap = pickle.load(f)

            with open(term_freq_cache_path, "rb") as f:
                self.term_frequencies = pickle.load(f)

            with open(doc_lengths_path, "rb") as f:
                self.doc_lengths = pickle.load(f)
        except Exception as e:
            raise Exception(f"Failed to load index from cache: {e}")

    def get_tf(self, doc_id: int, term: str) -> int:
        if doc_id not in self.term_frequencies.keys():
            raise ValueError("Invalid doc id")

        if term in self.term_frequencies[doc_id].keys():
            return self.term_frequencies[doc_id][term]
        else:
            return 0

    def get_bm25_idf(self, term: str) -> float:
        term = sanitise_term(term)

        N = len(self.docmap)
        df = len(self.get_documents(term))

        IDF = math.log((N - df + 0.5) / (df + 0.5) + 1)

        return IDF

    def get_bm25_tf(self, doc_id: int, term: str, k1: float, b: float) -> float:
        doc_length = self.doc_lengths[doc_id]
        avg_doc_length = self.__get_avg_doc_length()

        length_norm = 1 - b + b * (doc_length / avg_doc_length)

        tf = self.get_tf(doc_id, term)
        bm25_tf = (tf * (k1 + 1)) / (tf + k1 * length_norm)

        return bm25_tf
    
    def __get_avg_doc_length(self) -> float:
        n = len(self.doc_lengths.keys())
        s = sum(self.doc_lengths.values())

        if n > 0:
            return s / n
        else:
            return 0.0
