"""Rank-eight update on the last attention output projection; alpha/r = 1."""
import numpy as np
from .gpt2 import gelu

def normalized(x, scale, bias):
    c = x-x.mean(-1,keepdims=True)
    inv = 1/np.sqrt((c*c).mean(-1,keepdims=True)+1e-5)
    u = c*inv
    return u*scale+bias,(u,inv,scale)

def norm_backward(d, cache):
    u,inv,scale = cache
    d = d*scale
    return (d-d.mean(-1,keepdims=True)-u*(d*u).mean(-1,keepdims=True))*inv

class Adapter:
    def __init__(self, model):
        self.model = model
        self.a = np.random.default_rng(7).normal(0,.02,(768,8)).astype("f")
        self.b = np.zeros((8,768),"f")
        self.rows = model.embedding[model.tag_ids].copy()
        self.weight = model.w["h.11.attn.c_proj.weight"]
        self.merged = False

    def forward(self, residual, attended):
        w = self.model.w
        x = residual + attended@self.weight + w["h.11.attn.c_proj.bias"]
        if not self.merged:
            x += (attended@self.a)@self.b
        n,c1 = normalized(x,w["h.11.ln_2.weight"],w["h.11.ln_2.bias"])
        z = n@w["h.11.mlp.c_fc.weight"]+w["h.11.mlp.c_fc.bias"]
        y = x+gelu(z)@w["h.11.mlp.c_proj.weight"]+w["h.11.mlp.c_proj.bias"]
        h,c2 = normalized(y,w["ln_f.weight"],w["ln_f.bias"])
        self.cache = attended,z,h,c1,c2
        return h@self.rows.T

    def gradients(self, d):
        w = self.model.w
        attended,z,h,c1,c2 = self.cache
        dy = norm_backward(d@self.rows,c2)
        tanh = np.tanh(.79788456*(z+.044715*z**3))
        derivative = .5*(1+tanh)+.5*z*(1-tanh*tanh)*.79788456*(1+.134145*z*z)
        dn = ((dy@w["h.11.mlp.c_proj.weight"].T)*derivative)@w["h.11.mlp.c_fc.weight"].T
        dx = dy+norm_backward(dn,c1)
        return [attended.T@(dx@self.b.T),(attended@self.a).T@dx,d.T@h]
