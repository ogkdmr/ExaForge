#!/usr/bin/env python3
"""Filter model_cards_0000.jsonl to keep only entries where 'response' does NOT contain 'NO_MODEL_FOUND'.

Reads the original file, writes successful extractions to a new file.
The original file is left untouched.
"""

import json
from pathlib import Path
import sys

INPUT_FILE = sys.argv[1]
OUTPUT_FILE = sys.argv[2]
PATTERN = sys.argv[3]

def main():
    total = 0
    kept = 0
    skipped = 0

    with open(INPUT_FILE, "r") as fin, open(OUTPUT_FILE, "w") as fout:
        for line in fin:
            total += 1
            record = json.loads(line)
            if PATTERN in record.get("response", ""):
                skipped += 1
                continue
            fout.write(line)
            kept += 1

    print(f"Input file : {INPUT_FILE}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Total entries     : {total:,}")
    print(f"Kept (success)    : {kept:,}")
    print(f"Skipped {PATTERN}: {skipped:,}")


if __name__ == "__main__":
    main()
