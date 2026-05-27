#!/usr/bin/env python3
"""Load NWT Bible verses into rhypedb."""

import json
import sys
import time
import requests

HOST = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:4200"
DATA = "/home/joe/dev/nwt_verses.json"
BATCH_SIZE = 100

def query(q):
    resp = requests.post(f"{HOST}/query", json={"query": q})
    return resp.json()

def escape(s):
    return s.replace('\\', '\\\\').replace('"', '\\"')

def main():
    with open(DATA) as f:
        verses = json.load(f)

    limit = int(sys.argv[2]) if len(sys.argv) > 2 else len(verses)
    verses = verses[:limit]

    print(f"Loading {len(verses)} verses into {HOST}")
    start = time.time()
    errors = 0

    for i, v in enumerate(verses):
        text = escape(v["text"])
        book = escape(v["book_name"])
        q = f'Verse.create({{ book_name: "{book}", book_number: {v["book_number"]}, chapter: {v["chapter"]}, verse: {v["verse"]}, text: "{text}" }})'

        result = query(q)
        if "error" in result and result["error"]:
            errors += 1
            if errors <= 5:
                print(f"  ERROR on verse {i}: {result['error']}")

        if (i + 1) % BATCH_SIZE == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            print(f"  {i+1}/{len(verses)} ({rate:.0f} verses/sec)")

    elapsed = time.time() - start
    print(f"Done: {len(verses)} verses loaded in {elapsed:.1f}s ({len(verses)/elapsed:.0f}/sec), {errors} errors")

if __name__ == "__main__":
    main()
