from triton.tools.compile import desc
import json
import re

import numpy as np

from sentence_transformers import SentenceTransformer
from .constants import CACHE_DIR, DATA_DIR


class SemanticSearch:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
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

    def search(self, query: str, limit: int):
        if not self.embeddings.any():
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )

        query_embedding = self.generate_embedding(query)

        similarities = []

        for embedding, document in zip(self.embeddings, self.documents):
            similarity = cosine_similarity(embedding, query_embedding)
            similarities.append((similarity, document))

        similarities = sorted(similarities, key=lambda x: x[0], reverse=True)

        return similarities[:limit]


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents

        self.document_map = {}

        chunks: list[str] = []
        metadata: list[dict] = []

        movie_idx = 0

        for document in documents:
            self.document_map[document["id"]] = document

            if document["description"] == "":
                movie_idx += 1
                continue

            description_chunks = semantic_chunk(document["description"], 4, 1)
            chunk_idx = 0
            for c in description_chunks:
                chunks.append(c)
                metadata.append(
                    {
                        "movie_idx": movie_idx,
                        "chunk_idx": chunk_idx,
                        "total_chunks": len(description_chunks),
                    }
                )
                chunk_idx += 1

            movie_idx += 1

        self.chunk_embeddings = self.model.encode(chunks)
        self.chunk_metadata = metadata

        embeddings_dir = CACHE_DIR / "chunk_embeddings.npy"
        metadata_dir = CACHE_DIR / "chunk_metadata.json"

        with open(embeddings_dir, "wb") as f:
            np.save(f, self.chunk_embeddings)

        with open(metadata_dir, "w") as f:
            json.dump(
                {"chunks": self.chunk_metadata, "total_chunks": len(chunks)},
                f,
                indent=2,
            )

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents

        self.document_map = {}

        for document in documents:
            self.document_map[document["id"]] = document

        embeddings_dir = CACHE_DIR / "chunk_embeddings.npy"
        metadata_dir = CACHE_DIR / "chunk_metadata.json"

        if embeddings_dir.exists() and metadata_dir.exists():
            with open(embeddings_dir, "rb") as f:
                self.chunk_embeddings = np.load(f)

            with open(metadata_dir, "r") as f:
                self.chunk_metadata = json.load(f)["chunks"]

            return self.chunk_embeddings
        else:
            return self.build_chunk_embeddings(documents)


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
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def embed_query_text(query: str):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(query)

    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def search(query: str, limit: int):
    ss = SemanticSearch()

    movies_dir = DATA_DIR / "movies.json"

    with open(movies_dir, "r") as f:
        movies = json.load(f)

    documents = []
    for movie in movies["movies"]:
        documents.append(movie)

    embeddings = ss.load_or_create_embeddings(documents)

    results = ss.search(query, limit)

    count = 1
    for pair in results:
        score = pair[0]
        movie = pair[1]

        print(
            f"{count}. {movie['title']} (score: {score:.4f})\n{movie['description'][:100]}...\n"
        )

        count += 1


def chunk(text: str, chunk_size: int, overlap: int):
    words = text.split()
    step = chunk_size - overlap

    chunks = [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), step)]

    print(f"Chunking {len(text)} characters")
    for i, c in enumerate(chunks, 1):
        print(f"{i}. {c}")


def semantic_chunk(text: str, chunk_size: int, overlap: int, verbose: bool = False):
    sentences = re.split(string=text, pattern=r"(?<=[.!?])\s+")
    sentences = [s for s in sentences if s.strip()]
    step = chunk_size - overlap

    chunks = []
    for i in range(0, len(sentences), step):
        chunks.append(" ".join(sentences[i : i + chunk_size]))
        if i + chunk_size >= len(sentences):
            break

    if verbose:
        print(f"Semantically chunking {len(text)} characters")
        for i, c in enumerate(chunks, 1):
            print(f"{i}. {c}")

    return chunks


def embed_chunks():
    css = ChunkedSemanticSearch()

    movies_dir = DATA_DIR / "movies.json"

    with open(movies_dir, "r") as f:
        movies = json.load(f)

    documents = []
    for movie in movies["movies"]:
        documents.append(movie)

    embeddings = css.load_or_create_chunk_embeddings(documents)

    print(f"Generated {len(embeddings)} chunked embeddings")
