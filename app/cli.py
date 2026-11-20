import argparse
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from .core import LABELS, ROOT
from .gpt2 import GPT2
from .lora import Adapter

def load_model(merge=True):
    model = GPT2()
    head = Adapter(model)
    with np.load(ROOT/"adapter.npz") as weights:
        head.a,head.b,head.rows = [weights[k] for k in ("a","b","rows")]
    model.embedding[model.tag_ids] = head.rows
    if merge:
        head.weight = head.weight + head.a@head.b
        head.merged = True
    return head

def tag(head, text):
    logits = head.forward(*head.model.features(text))
    return LABELS[int(np.argmax(logits))]

def main():
    parser = argparse.ArgumentParser(description="Tag one legal clause or one clause per line.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text")
    group.add_argument("--file",type=Path)
    args = parser.parse_args()
    with threadpool_limits(2):
        try:
            lines = [args.text] if args.text is not None else args.file.read_text(encoding="utf-8-sig").splitlines()
            if not any(line.strip() for line in lines):
                parser.error("Supply at least one non-empty clause.")
            head = load_model()
            for i,text in enumerate(lines,1):
                if text.strip():
                    print(json.dumps({"line":i,"tag":tag(head,text),"text":text},ensure_ascii=True))
        except (ValueError,OSError) as exc:
            parser.error(str(exc))

if __name__ == "__main__":
    main()
