#!/usr/bin/env python3

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import math

from search_core.tokenise import sanitise_text, get_stopwords, sanitise_term
from search_core.index import InvertedIndex
from search_core.constants import BM25_K1, BM25_B

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

    tfidf_parser = subparsers.add_parser("tfidf", help="Get the tfidf of a term")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Token term")

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    bm25_tf_parser = subparsers.add_parser(
    "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 b parameter")

    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument("limit", type=int, nargs='?', default=5, help="Limit number of results to")
    bm25search_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25search_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 b parameter")


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

        case "tfidf":
            term = sanitise_term(args.term)

            index = InvertedIndex()
            index.load()

            tf = index.get_tf(args.doc_id, term)

            total_doc_count = len(index.docmap)
            term_match_doc_count = len(index.get_documents(term))

            idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1))

            tfidf = tf * idf

            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tfidf:.2f}")

        case "bm25idf":
            term = sanitise_term(args.term)

            index = InvertedIndex()
            index.load()

            bm25idf = index.get_bm25_idf(term)

            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")

        case "bm25tf":
            term = sanitise_term(args.term)

            index = InvertedIndex()
            index.load()

            bm25tf = index.get_bm25_tf(args.doc_id, term, args.k1, args.b)

            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")

        case "bm25search":
            index = InvertedIndex()
            index.load()

            results = index.bm25_search(args.query, args.limit, args.k1, args.b)

            count = 1
            for result in results:
                print(f"{count}. ({result[0]}) {index.docmap[result[0]]["title"]} - Score: {result[1]:.2f}")
        
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
