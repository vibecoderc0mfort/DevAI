"""Готовит обучающие данные для Socket 1.0 из стандартной библиотеки Python (лицензия PSF).

Каждый пример: «### Task: <описание из docstring>\\n### Code:\\n<код функции без docstring>\\n» + конец.
Код перепечатывается через ast.unparse, поэтому оформление у всех примеров одинаковое.
"""
import ast, glob, hashlib, os, sys
import numpy as np
from model import encode, EOS

MAX_CODE = 450          # символов кода в одном примере
MAX_TASK = 160


SKIP = ("site-packages", "dist-packages", "/test", "/docs", "/examples", "/scripts", "/locale", "idlelib", "lib2to3", "/benchmarks")


def collect(roots):
    files = []
    for root in roots:
        files += sorted(glob.glob(os.path.join(root, "**", "*.py"), recursive=True))
    files = [f for f in files if not any(x in f for x in SKIP)]
    seen, out = set(), []
    for f in files:
        try:
            src = open(f, encoding="utf-8").read()
            tree = ast.parse(src)
        except Exception:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            doc = ast.get_docstring(node)
            if not doc:
                continue
            first = " ".join(doc.strip().split("\n\n")[0].split())
            if len(first) > MAX_TASK and ". " in first:
                first = first.split(". ")[0] + "."          # берём первое предложение
            first = first.strip()
            if not (12 <= len(first) <= MAX_TASK) or not first.isascii():
                continue
            body = node.body[1:] or [ast.Pass()]
            clone = ast.FunctionDef(name=node.name, args=node.args, body=body, decorator_list=node.decorator_list,
                                    returns=node.returns, type_comment=None, type_params=getattr(node, "type_params", []))
            if isinstance(node, ast.AsyncFunctionDef):
                clone = ast.AsyncFunctionDef(name=node.name, args=node.args, body=body, decorator_list=node.decorator_list,
                                             returns=node.returns, type_comment=None, type_params=getattr(node, "type_params", []))
            try:
                code = ast.unparse(ast.fix_missing_locations(clone))
            except Exception:
                continue
            if not (40 <= len(code) <= MAX_CODE) or not code.isascii():
                continue
            h = hashlib.md5(code.encode()).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            out.append((first, code))
    return out


def to_tokens(first, code):
    return encode(f"### Task: {first}\n### Code:\n{code}\n") + [EOS]


if __name__ == "__main__":
    roots = sys.argv[1:] or ["/usr/lib/python3.12"]
    pairs = collect(roots)
    rng = np.random.default_rng(0)
    order = rng.permutation(len(pairs))
    n_val = max(50, len(pairs) // 50)
    val = [pairs[i] for i in order[:n_val]]
    train = [pairs[i] for i in order[n_val:]]
    os.makedirs("/home/claude/socket_data", exist_ok=True)
    for name, ds in (("train", train), ("val", val)):
        toks = [np.array(to_tokens(a, b), dtype=np.int16) for a, b in ds]
        lens = np.array([len(t) for t in toks])
        np.savez(f"/home/claude/socket_data/{name}.npz", tokens=np.concatenate(toks), lens=lens)
        print(name, "примеров:", len(ds), "токенов:", int(lens.sum()), "средняя длина:", int(lens.mean()), "макс:", int(lens.max()))
    import json
    json.dump([{"task": a, "code": b} for a, b in train[:5]], open("/tmp/samples.json", "w"), ensure_ascii=False, indent=1)
    print("\n--- пример ---\n### Task:", train[0][0], "\n### Code:\n" + train[0][1])
