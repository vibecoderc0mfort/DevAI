"""Обучение Socket 1.0 на NumPy. Запуск: python train.py --minutes 50 --out /путь/к/папке"""
import argparse, os, time, json
import numpy as np
from model import Config, init_params, forward, backward, n_params, generate, encode, decode, EOS

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/home/claude/socket_data")
ap.add_argument("--out", default="/home/claude/socket_out")
ap.add_argument("--minutes", type=float, default=50)
ap.add_argument("--bench", action="store_true")
ap.add_argument("--lr", type=float, default=3e-3)
ap.add_argument("--B", type=int, default=4)
ap.add_argument("--T", type=int, default=512)
ap.add_argument("--resume", default="")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

cfg = Config(d=128, layers=4, heads=4, ff=384, ctx=1024)
p = init_params(cfg, seed=0)
if args.resume:
    z = np.load(args.resume)
    p = {k: z[k] for k in p}
print("параметров:", n_params(p), flush=True)


def load(name):
    z = np.load(f"{args.data}/{name}.npz")
    toks, lens = z["tokens"].astype(np.int64), z["lens"]
    offs = np.concatenate([[0], np.cumsum(lens)])
    return [toks[offs[i]:offs[i + 1]] for i in range(len(lens))]


train, val = load("train"), load("val")
rng = np.random.default_rng(1)


def batches(samples, B, T, rng, shuffle=True):
    """Каждая строка начинается с начала примера и добивается следующими до длины T+1."""
    while True:
        order = rng.permutation(len(samples)) if shuffle else np.arange(len(samples))
        i = 0
        while i < len(order):
            rows = []
            for _ in range(B):
                row = []
                while len(row) < T + 1 and i < len(order):
                    row.extend(samples[order[i]].tolist()); i += 1
                if len(row) < T + 1:
                    row += [EOS] * (T + 1 - len(row))
                rows.append(row[:T + 1])
            arr = np.array(rows, dtype=np.int64)
            yield arr[:, :-1], arr[:, 1:]
        if not shuffle:
            return


def evaluate(p, n_rows=24):
    tot, cnt = 0.0, 0
    for x, y in batches(val, 4, args.T, None, shuffle=False):
        l, _ = forward(p, x, cfg, y, keep=False)
        tot += float(l); cnt += 1
        if cnt * 4 >= n_rows:
            break
    return tot / cnt


m = {k: np.zeros_like(v) for k, v in p.items()}
v2 = {k: np.zeros_like(v) for k, v in p.items()}
b1, b2, eps, wd = 0.9, 0.95, 1e-8, 0.05
budget = args.minutes * 60
gen = batches(train, args.B, args.T, rng)
t0 = time.time(); step = 0; tokens = 0; ema = None; best = 1e9
log = open(f"{args.out}/train.log", "a")


def say(msg):
    print(msg, flush=True); log.write(msg + "\n"); log.flush()


PROMPT = "### Task: Return the sum of two numbers.\n### Code:\n"
while True:
    el = time.time() - t0
    frac = el / budget
    if frac >= 1.0 or (args.bench and step >= 6):
        break
    warm = min(1.0, (step + 1) / 80)
    lr = args.lr * warm * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * frac)))
    x, y = next(gen)
    loss, cache = forward(p, x, cfg, y)
    g = backward(p, cache, cfg)
    gn = np.sqrt(sum(float((a * a).sum()) for a in g.values()))
    clip = min(1.0, 1.0 / (gn + 1e-6))
    step += 1; tokens += x.size
    for k in p:
        gk = g[k] * clip
        m[k] = b1 * m[k] + (1 - b1) * gk
        v2[k] = b2 * v2[k] + (1 - b2) * gk * gk
        mh = m[k] / (1 - b1 ** step); vh = v2[k] / (1 - b2 ** step)
        upd = mh / (np.sqrt(vh) + eps)
        if p[k].ndim == 2 and k != "E":
            upd = upd + wd * p[k]
        p[k] -= (lr * upd).astype(p[k].dtype)
    ema = float(loss) if ema is None else 0.98 * ema + 0.02 * float(loss)
    if args.bench:
        print(f"шаг {step}: loss {float(loss):.3f}, {tokens / (time.time() - t0):.0f} токенов/с", flush=True)
    if step % 50 == 0:
        say(f"шаг {step} | {el / 60:.1f} мин | токенов {tokens / 1e6:.2f}M | lr {lr:.5f} | loss {ema:.3f} | {tokens / el:.0f} ток/с")
    if step % 250 == 0:
        vl = evaluate(p)
        say(f"   проверка: val loss {vl:.3f}")
        np.savez(f"{args.out}/last.npz", **p)
        if vl < best:
            best = vl; np.savez(f"{args.out}/best.npz", **p)
        out = generate(p, cfg, encode(PROMPT), max_new=120, temperature=0.0)
        say("   пример: " + decode(out[len(encode(PROMPT)):]).replace("\n", "\\n")[:200])

if not args.bench:
    vl = evaluate(p, 48)
    say(f"ГОТОВО: шагов {step}, токенов {tokens / 1e6:.2f}M, итоговый val loss {vl:.3f}")
    np.savez(f"{args.out}/last.npz", **p)
    if vl <= best:
        np.savez(f"{args.out}/best.npz", **p)
    json.dump(cfg.to_dict(), open(f"{args.out}/config.json", "w"))
