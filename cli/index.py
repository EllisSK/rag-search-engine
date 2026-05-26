import pickle
import json

from pathlib import Path, PosixPath

from tokenise import sanitise_text, get_stopwords

class InvertedIndex:
    def __init__(self):
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, Path] = {}
        self.stopwords = get_stopwords()

    def __add_document(self, doc_id: int, text: str):
        tokenised_text = sanitise_text(text, self.stopwords)
        for token in tokenised_text:
            if token in self.index.keys():
                self.index[token].add(doc_id)
            else:
                self.index[token] = {doc_id}

    def get_documents(self, term: str):
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

        for movie in movie_data["movies"]:
            doc_id = movie["id"]
            self.docmap[doc_id] = movie
            self.__add_document(doc_id, f"{movie['title']} {movie['description']}")

    def save(self):
        index_cache_path = Path("cache/index.pkl")
        docmap_cache_path = Path("cache/docmap.pkl")

        index_cache_path.parent.mkdir(parents=True, exist_ok=True)

        with open(index_cache_path, "wb") as f:
            pickle.dump(self.index, f)
        
        with open(docmap_cache_path, "wb") as f:
            pickle.dump(self.docmap, f)

    def load(self):
        index_cache_path = Path("cache/index.pkl")
        docmap_cache_path = Path("cache/docmap.pkl")

        try:
            with open(index_cache_path, "rb") as f:
                self.index = pickle.load(f)
            
            with open(docmap_cache_path, "rb") as f:
                self.docmap = pickle.load(f)
        except Exception as e:
            raise Exception(f"Failed to load index from cache: {e}")

