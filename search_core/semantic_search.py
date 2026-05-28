import json

import numpy as np

from sentence_transformers import SentenceTransformer
from .constants import CACHE_DIR, DATA_DIR

class SemanticSearch:
    def __init__(self) -> None:
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents: list[dict] | None = None
        self.document_map: dict | None = None

    def generate_embedding(self, text: str):
        if not text.strip():
            raise ValueError("Bad text")
        
        embedding = self.model.encode([text])

        return embedding[0]
    
    def build_embeddings(self, documents: list[dict]):
        self.documents = documents

        self.document_map = {}
        str_reps = []

        for document in documents:
            self.document_map[document["id"]] = document
            str_reps.append(f"{document['title']}: {document['description']}")

        self.embeddings = self.model.encode(str_reps, show_progress_bar=True)

        embeddings_dir = CACHE_DIR / "movie_embeddings.npy" 

        with open(embeddings_dir, "wb") as f:
            np.save(f, self.embeddings)
        
        return self.embeddings
    
    def load_or_create_embeddings(self, documents: list[dict]):
        self.documents = documents

        self.document_map = {}
        str_reps = []

        for document in documents:
            self.document_map[document["id"]] = document

        embeddings_dir = CACHE_DIR / "movie_embeddings.npy" 

        if embeddings_dir.is_file():
            with open(embeddings_dir, "rb") as f:
                embeddings = np.load(f)

                if len(embeddings) == len(documents):
                    self.embeddings = embeddings
                else:
                    self.embeddings = self.build_embeddings(documents)
        else:
            self.embeddings = self.build_embeddings(documents)

        return self.embeddings

def verify_model():

    ss = SemanticSearch()

    print(f"Model loaded: {ss.model}")
    print(f"Max sequence length: {ss.model.max_seq_length}")

def embed_text(text: str):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    ss = SemanticSearch()

    movies_dir = DATA_DIR / "movies.json"

    with open(movies_dir, "r") as f:
        movies = json.load(f)

    documents = []
    for movie in movies["movies"]:
        documents.append(movie)

    embeddings = ss.load_or_create_embeddings(documents)

    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")
