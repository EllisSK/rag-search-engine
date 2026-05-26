#!/usr/bin/env python3

import argparse
import json
import string
import math

from pathlib import Path

from tokenise import sanitise_text, get_stopwords, sanitise_term
from index import InvertedIndex

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build the inverted index and save it to disk")

    tf_parser = subparsers.add_parser("tf", help="Get the term frequency of a single token")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Token term")

    idf_parser = subparsers.add_parser("idf", help="Get the inverse document frequency of a term")
    idf_parser.add_argument("term", type=str, help="Token term")

    args = parser.parse_args()

    match args.command:
        case "search":
            index = InvertedIndex()
            index.load()

            print(f"Searching for: {args.query}")

            stopwords = get_stopwords()

            query_tokens = sanitise_text(args.query, stopwords)

            results = []
            for token in query_tokens:
                try:
                    result = index.get_documents(token)
                    for r in result:
                        results.append(r)
                        if len(results) == 5:
                            break
                except:
                    pass

            if results:
                count = 1
                for result in results:
                    title = index.docmap[result]["title"]
                    print(f"{count}. {title}")
                    count += 1
            else:
                print("No results found")

        case "build":
            index = InvertedIndex()
            index.build()
            index.save()

        case "tf":
            term = sanitise_term(args.term)

            index = InvertedIndex()
            index.load()

            freq = index.get_tf(args.doc_id, term)

            if freq == 0:
                print(0)
            elif freq == 1:
                print(f"{args.term} appears {freq} time in document {args.doc_id}")
            else:
                print(f"{args.term} appears {freq} times in document {args.doc_id}")

        case "idf":
            term = sanitise_term(args.term)

            index = InvertedIndex()
            index.load()

            total_doc_count = len(index.docmap)
            term_match_doc_count = len(index.get_documents(term))

            idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1))

            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
