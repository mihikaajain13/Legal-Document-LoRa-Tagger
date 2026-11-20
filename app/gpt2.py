"""CPU forward pass for the pinned GPT-2 small safetensors checkpoint."""
import json
import struct
import numpy as np
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
from .core import LABELS

REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"

def asset(name):
    args = dict(repo_id="gpt2", filename=name, revision=REVISION)
    try:
        return hf_hub_download(**args, local_files_only=True)
    except FileNotFoundError:
        return hf_hub_download(**args)

def norm(x, scale, bias):
    centered = x - x.mean(-1, keepdims=True)
    return centered / np.sqrt((centered**2).mean(-1, keepdims=True)+1e-5)*scale+bias

def gelu(x):
    return .5*x*(1+np.tanh(.79788456*(x+.044715*x**3)))

class GPT2:
    def __init__(self):
        path = asset("model.safetensors")
        with open(path,"rb") as fp:
            length = struct.unpack("<Q",fp.read(8))[0]
            header = json.loads(fp.read(length))
        self.w = {k: np.memmap(path,dtype="<f4",mode="r",shape=v["shape"],
                               offset=8+length+v["data_offsets"][0])
                  for k,v in header.items() if k!="__metadata__" and v["dtype"]=="F32"}
        self.tok = Tokenizer.from_file(asset("tokenizer.json"))
        # Seed the new tied input/output rows with familiar semantic anchors.
        anchors = [" payment"," confidential"," termination"," audit"]
        rows = [self.w["wte.weight"][self.tok.encode(s).ids].mean(0) for s in anchors]
        self.tag_ids = []
        for label in LABELS:
            tag = "<"+label+">"
            self.tok.add_special_tokens([tag])
            self.tag_ids.append(self.tok.token_to_id(tag))
        self.embedding = np.vstack([self.w["wte.weight"],rows])

    def features(self, text):
        ids = self.tok.encode(text.strip()).ids
        if not ids or len(ids)>256:
            raise ValueError("Supply a clause of 1 to 256 GPT-2 tokens.")
        x = self.embedding[ids] + self.w["wpe.weight"][:len(ids)]
        mask = np.triu(np.ones((len(ids),len(ids)),bool),1)
        for layer in range(12):
            prefix = f"h.{layer}."
            def w(k):
                return self.w[prefix+k]
            n = norm(x,w("ln_1.weight"),w("ln_1.bias"))
            q,k,v = np.split(n@w("attn.c_attn.weight")+w("attn.c_attn.bias"),3,axis=-1)
            def heads(matrix):
                return matrix.reshape(len(ids),12,64).transpose(1,0,2)
            score = np.where(mask,-1e9,heads(q)@heads(k).transpose(0,2,1)/8)
            probs = np.exp(score-score.max(-1,keepdims=True))
            attended = ((probs/probs.sum(-1,keepdims=True))@heads(v)).transpose(1,0,2).reshape(len(ids),768)
            if layer==11:
                return x[-1],attended[-1]
            x = x + attended@w("attn.c_proj.weight")+w("attn.c_proj.bias")
            n = norm(x,w("ln_2.weight"),w("ln_2.bias"))
            x += gelu(n@w("mlp.c_fc.weight")+w("mlp.c_fc.bias"))@w("mlp.c_proj.weight")+w("mlp.c_proj.bias")
