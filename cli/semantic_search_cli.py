#!/usr/bin/env python3

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse

from search_core.semantic_search import verify_model, embed_text

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify", help="Verify model info")

    embedding_parser = subparsers.add_parser("embed_text", help="Embed some text")
    embedding_parser.add_argument("text", type=str, help="Text to embed")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()

        case "embed_text":
            embed_text(args.text)

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()