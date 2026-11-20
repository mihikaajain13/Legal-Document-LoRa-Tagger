import numpy as np
from threadpoolctl import threadpool_limits
from .core import LABELS, ROOT, records
from .gpt2 import GPT2
from .lora import Adapter

def train():
    model = GPT2()
    head = Adapter(model)
    rows = records("train")
    x,v = map(np.stack,zip(*(model.features(row["text"]) for row in rows)))
    y = np.array([LABELS.index(row["label"]) for row in rows])
    params = [head.a,head.b,head.rows]
    m = [np.zeros_like(p) for p in params]
    s = [np.zeros_like(p) for p in params]
    rng = np.random.default_rng(12)
    for step in range(1,181):
        idx = rng.choice(len(rows),12,replace=False)
        logits = head.forward(x[idx],v[idx])
        probs = np.exp(logits-logits.max(-1,keepdims=True))
        probs /= probs.sum(-1,keepdims=True)
        probs[np.arange(len(idx)),y[idx]] -= 1
        for i,(p,g) in enumerate(zip(params,head.gradients(probs/len(idx)))):
            m[i] = .9*m[i]+.1*g
            s[i] = .999*s[i]+.001*g*g
            p -= .0006*(m[i]/(1-.9**step))/(np.sqrt(s[i]/(1-.999**step))+1e-8)
    np.savez_compressed(ROOT/"adapter.npz",a=head.a,b=head.b,rows=head.rows)

if __name__ == "__main__":
    with threadpool_limits(2):
        train()
