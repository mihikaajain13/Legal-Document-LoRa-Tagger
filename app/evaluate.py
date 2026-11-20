import argparse
import json
import time
import numpy as np
from threadpoolctl import threadpool_limits
from .cli import load_model,tag
from .core import ROOT,LABELS,records
from .lora import Adapter

def evaluate(manual_seconds=None):
    head = load_model(False)
    base = Adapter(head.model)
    # Restore original semantic anchor rows for the frozen baseline.
    base.rows = np.stack([head.model.w["wte.weight"][head.model.tok.encode(s).ids].mean(0)
                          for s in (" payment"," confidential"," termination"," audit")])
    rows = records("test")
    features = [head.model.features(r["text"]) for r in rows]
    original = np.stack([base.forward(*f) for f in features])
    active = np.stack([head.forward(*f) for f in features])
    head.weight = head.weight+head.a@head.b
    head.merged = True
    merged = np.stack([head.forward(*f) for f in features])
    np.testing.assert_allclose(active,merged,atol=.001,rtol=.001)
    start = time.perf_counter()
    preds = [tag(head,r["text"]) for r in rows]
    elapsed = time.perf_counter()-start
    truth = np.array([LABELS.index(r["label"]) for r in rows])
    confusion = np.zeros((len(LABELS),len(LABELS)),int)
    np.add.at(confusion,(truth,[LABELS.index(p) for p in preds]),1)
    f1 = 2*confusion.diagonal()/(confusion.sum(0)+confusion.sum(1)).clip(1)
    result = dict(labels=LABELS,confusion=confusion.tolist(),correct=int(confusion.trace()),
                  total=len(rows),accuracy=float(confusion.trace()/len(rows)),macro_f1=float(f1.mean()),
                  baseline_accuracy=float((original.argmax(1)==truth).mean()),
                  adapter_parameters=int(head.a.size+head.b.size), new_embedding_parameters=int(head.rows.size),
                  warm_seconds=elapsed,clauses_per_second=len(rows)/elapsed,
                  max_merge_error=float(np.abs(active-merged).max()))
    if manual_seconds:
        result["observed_speed_ratio_against_supplied_manual_time"] = manual_seconds/elapsed
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual-seconds",type=float,help="Measured total human time on the same 12 clauses")
    args = parser.parse_args()
    if args.manual_seconds is not None and args.manual_seconds<=0:
        parser.error("Manual duration must be positive")
    with threadpool_limits(2):
        print(json.dumps(evaluate(args.manual_seconds),indent=2))
