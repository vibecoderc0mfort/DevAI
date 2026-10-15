import numpy as np
from model import Config, init_params, forward, backward

cfg = Config(d=8, layers=2, heads=2, ff=12, ctx=16, vocab=11)
p = {k: v.astype(np.float64) for k, v in init_params(cfg, seed=1).items()}
rng = np.random.default_rng(2)
for k in p:                      # чтобы градиенты были заметными
    p[k] = p[k] + rng.standard_normal(p[k].shape) * 0.3
tok = rng.integers(0, 11, size=(2, 6))
tgt = rng.integers(0, 11, size=(2, 6))
loss, cache = forward(p, tok, cfg, tgt)
g = backward(p, cache, cfg)
worst = 0.0
for k in p:
    flat = p[k].reshape(-1)
    for idx in rng.choice(flat.size, size=min(6, flat.size), replace=False):
        old = flat[idx]; eps = 1e-6
        flat[idx] = old + eps; lp, _ = forward(p, tok, cfg, tgt, keep=False)
        flat[idx] = old - eps; lm, _ = forward(p, tok, cfg, tgt, keep=False)
        flat[idx] = old
        num = (lp - lm) / (2 * eps)
        ana = g[k].reshape(-1)[idx]
        rel = abs(num - ana) / max(1e-8, abs(num) + abs(ana))
        worst = max(worst, rel)
        if rel > 1e-4:
            print("РАСХОЖДЕНИЕ", k, idx, num, ana, rel)
print("loss", float(loss), "| худшая относительная ошибка градиента:", worst)
print("ГРАДИЕНТЫ ВЕРНЫ" if worst < 1e-5 else "ЕСТЬ ОШИБКА")
