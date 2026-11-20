"""Timed human baseline on the same held-out clauses; labels hidden until the end."""
import random
import time
from .core import records,LABELS

def main():
    rows = records("test")
    random.Random(17).shuffle(rows)
    print("Labels:", ", ".join(LABELS))
    input("Press Enter when ready. Timing begins with the first clause.")
    start = time.perf_counter()
    correct = 0
    for row in rows:
        print(row["text"])
        answer = input("Tag: ").strip().upper()
        while answer not in LABELS:
            answer = input("Choose a listed tag: ").strip().upper()
        correct += answer == row["label"]
    seconds = time.perf_counter()-start
    print(f"{correct}/{len(rows)} correct in {seconds:.2f} seconds.")
    print(f"Compare using: python -m app.evaluate --manual-seconds {seconds:.2f}")

if __name__ == "__main__":
    main()
