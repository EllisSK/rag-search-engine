#!/usr/bin/env python3

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse

from search_core.semantic_search import verify_model, embed_text, verify_embeddings, embed_query_text, search, chunk

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify", help="Verify model info")

    embedding_parser = subparsers.add_parser("embed_text", help="Embed some text")
    embedding_parser.add_argument("text", type=str, help="Text to embed")

    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="Verifies the embeddings of the movies")

    embed_query_parser = subparsers.add_parser("embed_query", help="Embed a query")
    embed_query_parser.add_argument("query", type=str, help="The query to embed")

    search_parser = subparsers.add_parser("search", help="Search using a query")
    search_parser.add_argument("query", type=str, help="The query to search")
    search_parser.add_argument("--limit", type=int, default=5, nargs='?')

    chunk_parser = subparsers.add_parser("chunk", help="Split long text into smaller chunks")
    chunk_parser.add_argument("text", type=str, help="The text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, default=200, nargs='?')
    chunk_parser.add_argument("--overlap", type=int, default=0, nargs='?')

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()

        case "embed_text":
            embed_text(args.text)

        case "verify_embeddings":
            verify_embeddings()

        case "embed_query":
            embed_query_text(args.query)

        case "search":
            search(args.query, args.limit)

        case "chunk":
            chunk(args.text, args.chunk_size, args.overlap)


        case _:
            parser.print_help()

if __name__ == "__main__":
    main()