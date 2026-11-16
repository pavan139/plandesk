"""Command line: build the index, ask a question, run or re-score evals."""

from __future__ import annotations

import argparse
import json
import sys

from .corpus import CHUNKS_PATH


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="plandesk")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("build", help="extract chunks from data/raw PDFs")

    ask = sub.add_parser("ask", help="answer one question")
    ask.add_argument("question")
    ask.add_argument("--model", default="gpt-5.4-mini")
    ask.add_argument("--mode", default="full", choices=["full", "naive_rag", "closed_book"])

    ev = sub.add_parser("eval", help="run the eval set against the API")
    ev.add_argument("--model", default="gpt-5.4-mini")
    ev.add_argument("--mode", default="full", choices=["full", "naive_rag", "closed_book"])

    rs = sub.add_parser("rescore", help="re-score saved results offline")
    rs.add_argument("paths", nargs="+")

    sv = sub.add_parser("serve", help="run the local web demo")
    sv.add_argument("--port", type=int, default=8000)

    args = p.parse_args(argv)

    if args.cmd == "build":
        from .corpus import build_chunks, save_chunks

        chunks = build_chunks()
        save_chunks(chunks)
        print(f"wrote {len(chunks)} chunks to {CHUNKS_PATH}")
    elif args.cmd == "ask":
        from openai import OpenAI

        from .answer import answer
        from .retrieve import default_retriever

        retriever = None if args.mode == "closed_book" else default_retriever()
        res = answer(args.question, retriever, OpenAI(), model=args.model, mode=args.mode)
        print(res.answer)
        for c in res.citations:
            mark = "verified" if c.verified else "NOT FOUND in source"
            print(f"  - {c.source} [{c.id}] ({mark}): \"{c.quote}\"")
        if res.guard:
            print(f"  guard: {res.guard}")
    elif args.cmd == "eval":
        from .evaluate import run

        print(json.dumps(run(args.model, args.mode), indent=2))
    elif args.cmd == "rescore":
        from pathlib import Path

        from .evaluate import rescore

        for path in args.paths:
            print(json.dumps(rescore(Path(path)), indent=2))
    elif args.cmd == "serve":
        import uvicorn

        uvicorn.run("plandesk.web:app", port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
