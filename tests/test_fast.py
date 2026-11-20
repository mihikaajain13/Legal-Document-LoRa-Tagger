import unittest
import numpy as np
from app.core import LABELS,ROOT,records

class Checks(unittest.TestCase):
    def test_split(self):
        train,test = records("train"),records("test")
        self.assertFalse({r["text"] for r in train} & {r["text"] for r in test})
        self.assertEqual(set(LABELS),{r["label"] for r in test})

    def test_adapter(self):
        with np.load(ROOT/"adapter.npz") as p:
            self.assertEqual(p["a"].shape,(768,8))
            self.assertEqual(p["b"].shape,(8,768))
            self.assertEqual(p["rows"].shape,(4,768))
            self.assertGreater(np.linalg.norm(p["b"]),0)
