#!/usr/bin/env python3

import argparse
import json
import string

from pathlib import Path
from nltk.stem import PorterStemmer

def get_stopwords() -> list[str]: 
    stopwords_path = Path("data/stopwords.txt")
    
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

    text = text.split(" ")
    nw = []
    for token in text:
        if token != " " and token not in stopwords:
            nw.append(token)
    text = nw

    stemmer = PorterStemmer()
    st = []
    for token in text:
        st.append(stemmer.stem(token))
    text = st

    return text

def search(query: str) -> list[dict]:
    movie_path = "data/movies.json"

    with open(movie_path, "r") as f:
        movie_data = json.load(f)

    stopwords = get_stopwords()

    query = sanitise_text(query, stopwords)
    matched = []

    for movie in movie_data["movies"]:
        title = movie["title"]

        title = sanitise_text(title, stopwords)

        for title_token in title:
            for query_token in query:
                if query_token in title_token:
                    if movie not in matched:
                        matched.append(movie)


    return matched

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            results = search(args.query)
            num = 1

            if results:
                for movie in results:
                    print(f"{num}. {movie["title"]}")
                    num += 1
                    if num >= 6:
                        break
            else:
                print("No results found")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
