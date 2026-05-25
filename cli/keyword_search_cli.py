#!/usr/bin/env python3

import argparse
import json
import string

from pathlib import Path

def sanitise_text(text: str) -> str:
    text = text.lower()
    
    np = ""
    for char in text:
        if char not in string.punctuation:
            np += char

    text = np

    return text

def search(query: str) -> list[dict]:
    movie_path = "data/movies.json"

    with open(movie_path, "r") as f:
        movie_data = json.load(f)

    matched = []

    for movie in movie_data["movies"]:
        title = movie["title"]

        query = sanitise_text(query)
        title = sanitise_text(title)

        if query in title:
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
