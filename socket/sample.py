"""Пробная генерация: python sample.py weights.npz [температура] [top_k]"""
import sys, numpy as np
from model import Config, generate, encode, decode

cfg = Config(d=128, layers=4, heads=4, ff=384, ctx=1024)
z = np.load(sys.argv[1]); p = {k: z[k] for k in z.files}
temp = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
topk = int(sys.argv[3]) if len(sys.argv) > 3 else 20
PROMPTS = [
    "Return the sum of a list of numbers.",
    "Check whether a string is a palindrome.",
    "Read a text file and return its lines.",
    "Return the largest item in a list.",
    "Convert a string to lowercase.",
]
for i, t in enumerate(PROMPTS):
    prompt = encode(f"### Task: {t}\n### Code:\n")
    out = generate(p, cfg, prompt, max_new=260, temperature=temp, top_k=topk, seed=i)
    print(f"--- {t}\n{decode(out[len(prompt):])}\n")
