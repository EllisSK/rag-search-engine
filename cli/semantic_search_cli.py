#!/usr/bin/env python3

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse

from search_core.semantic_search import verify_model

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify", help="Verify model info")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()