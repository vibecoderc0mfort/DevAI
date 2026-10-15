"""Socket 1.0: маленький трансформер (в стиле Llama), написанный с нуля на NumPy.

Архитектура: байтовый словарь, RMSNorm, RoPE (парами соседних чисел), многоголовое внимание
с причинной маской, SwiGLU, общие веса входного и выходного слоя. Прямой и обратный проход
написаны вручную, без PyTorch и других фреймворков.
"""
import numpy as np

# Словарь: 0 <unk>, 1 <s>, 2 </s>, 3..258 байты 0..255, 259 «▁» (пробел)
V = 260
UNK, BOS, EOS, SPACE_ID = 0, 1, 2, 259


class Config:
    def __init__(self, d=128, layers=4, heads=4, ff=384, ctx=1024, vocab=V, base=10000.0, eps=1e-5):
        self.d, self.layers, self.heads, self.ff = d, layers, heads, ff
        self.ctx, self.vocab, self.base, self.eps = ctx, vocab, base, eps
        assert d % heads == 0 and (d // heads) % 2 == 0
        self.dh = d // heads

    def to_dict(self):
        return dict(d=self.d, layers=self.layers, heads=self.heads, ff=self.ff,
                    ctx=self.ctx, vocab=self.vocab, base=self.base, eps=self.eps)


def init_params(cfg, seed=0, dtype=np.float32):
    rng = np.random.default_rng(seed)
    n = lambda *s, std=0.02: (rng.standard_normal(s) * std).astype(dtype)
    p = {"E": n(cfg.vocab, cfg.d), "gf": np.ones(cfg.d, dtype)}
    out_std = 0.02 / np.sqrt(2 * cfg.layers)
    for i in range(cfg.layers):
        p[f"g1.{i}"] = np.ones(cfg.d, dtype)
        p[f"g2.{i}"] = np.ones(cfg.d, dtype)
        for name in ("Wq", "Wk", "Wv"):
            p[f"{name}.{i}"] = n(cfg.d, cfg.d)
        p[f"Wo.{i}"] = n(cfg.d, cfg.d, std=out_std)
        p[f"Wg.{i}"] = n(cfg.d, cfg.ff)
        p[f"Wu.{i}"] = n(cfg.d, cfg.ff)
        p[f"Wd.{i}"] = n(cfg.ff, cfg.d, std=out_std)
    return p


def n_params(p):
    return int(sum(v.size for v in p.values()))


# ---------------------------------------------------------------- вспомогательные функции

def mm(x, W):
    """(…, in) @ (in, out) через одну большую матрицу, так быстрее."""
    return (x.reshape(-1, x.shape[-1]) @ W).reshape(x.shape[:-1] + (W.shape[1],))


def rms_fwd(x, g, eps):
    r = 1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    xn = x * r
    return xn * g, (x, r, xn, g)


def rms_bwd(dy, c):
    x, r, xn, g = c
    dg = np.sum(dy * xn, axis=tuple(range(dy.ndim - 1)))
    u = dy * g
    dx = r * u - x * (r ** 3) * np.mean(u * x, axis=-1, keepdims=True)
    return dx, dg


def rope_tables(T, dh, base, dtype):
    inv = base ** (-np.arange(0, dh, 2) / dh)
    ang = np.outer(np.arange(T), inv)
    return np.cos(ang).astype(dtype), np.sin(ang).astype(dtype)


def rope(x, cos, sin, inverse=False):
    """x: (B,T,H,dh). Пары (x[2i], x[2i+1]) поворачиваются на угол pos*freq_i, как в llama.cpp."""
    B, T, H, dh = x.shape
    xp = x.reshape(B, T, H, dh // 2, 2)
    x0, x1 = xp[..., 0], xp[..., 1]
    c, s = cos[None, :T, None, :], sin[None, :T, None, :]
    if inverse:
        s = -s
    r0 = x0 * c - x1 * s
    r1 = x0 * s + x1 * c
    return np.stack([r0, r1], axis=-1).reshape(B, T, H, dh)


def silu(a):
    sg = 1.0 / (1.0 + np.exp(-a))
    return a * sg, sg


# ---------------------------------------------------------------- прямой проход

def forward(p, tokens, cfg, targets=None, keep=True):
    """tokens: (B,T) int. Возвращает (logits или loss, cache)."""
    B, T = tokens.shape
    dt = p["E"].dtype
    cos, sin = rope_tables(T, cfg.dh, cfg.base, dt)
    mask = np.triu(np.ones((T, T), dtype=bool), 1)
    scale = 1.0 / np.sqrt(cfg.dh)
    x = p["E"][tokens]
    layers = []
    for i in range(cfg.layers):
        h, c1 = rms_fwd(x, p[f"g1.{i}"], cfg.eps)
        q = mm(h, p[f"Wq.{i}"]).reshape(B, T, cfg.heads, cfg.dh)
        k = mm(h, p[f"Wk.{i}"]).reshape(B, T, cfg.heads, cfg.dh)
        v = mm(h, p[f"Wv.{i}"]).reshape(B, T, cfg.heads, cfg.dh)
        qr, kr = rope(q, cos, sin), rope(k, cos, sin)
        qh, kh, vh = (a.transpose(0, 2, 1, 3) for a in (qr, kr, v))
        s = (qh @ kh.transpose(0, 1, 3, 2)) * scale
        s = np.where(mask, -1e9, s)
        s = s - s.max(-1, keepdims=True)
        P = np.exp(s)
        P /= P.sum(-1, keepdims=True)
        o = (P @ vh).transpose(0, 2, 1, 3).reshape(B, T, cfg.d)
        x = x + mm(o, p[f"Wo.{i}"])
        h2, c2 = rms_fwd(x, p[f"g2.{i}"], cfg.eps)
        a = mm(h2, p[f"Wg.{i}"])
        b = mm(h2, p[f"Wu.{i}"])
        sa, sg = silu(a)
        m = sa * b
        x = x + mm(m, p[f"Wd.{i}"])
        if keep:
            layers.append((c1, h, qh, kh, vh, P, o, c2, h2, a, b, sa, sg, m))
    xf, cf = rms_fwd(x, p["gf"], cfg.eps)
    logits = mm(xf, p["E"].T)
    if targets is None:
        return logits, None
    z = logits - logits.max(-1, keepdims=True)
    lp = z - np.log(np.exp(z).sum(-1, keepdims=True))
    N = B * T
    loss = -lp[np.arange(B)[:, None], np.arange(T)[None, :], targets].mean()
    cache = (tokens, targets, cos, sin, lp, layers, cf, xf) if keep else None
    return loss, cache


# ---------------------------------------------------------------- обратный проход

def backward(p, cache, cfg):
    tokens, targets, cos, sin, lp, layers, cf, xf = cache
    B, T = tokens.shape
    N = B * T
    scale = 1.0 / np.sqrt(cfg.dh)
    g = {k: np.zeros_like(v) for k, v in p.items()}

    dlog = np.exp(lp)
    dlog[np.arange(B)[:, None], np.arange(T)[None, :], targets] -= 1.0
    dlog /= N
    d2 = dlog.reshape(-1, cfg.vocab)
    g["E"] += d2.T @ xf.reshape(-1, cfg.d)
    dxf = (d2 @ p["E"]).reshape(B, T, cfg.d)
    dx, g["gf"] = rms_bwd(dxf, cf)

    for i in reversed(range(cfg.layers)):
        c1, h, qh, kh, vh, P, o, c2, h2, a, b, sa, sg, m = layers[i]
        # MLP
        dy = dx
        g[f"Wd.{i}"] += m.reshape(-1, cfg.ff).T @ dy.reshape(-1, cfg.d)
        dm = mm(dy, p[f"Wd.{i}"].T)
        db = dm * sa
        da = dm * b * sg * (1.0 + a * (1.0 - sg))
        g[f"Wg.{i}"] += h2.reshape(-1, cfg.d).T @ da.reshape(-1, cfg.ff)
        g[f"Wu.{i}"] += h2.reshape(-1, cfg.d).T @ db.reshape(-1, cfg.ff)
        dh2 = mm(da, p[f"Wg.{i}"].T) + mm(db, p[f"Wu.{i}"].T)
        dxn, g[f"g2.{i}"] = rms_bwd(dh2, c2)
        dx = dx + dxn
        # внимание
        dy = dx
        g[f"Wo.{i}"] += o.reshape(-1, cfg.d).T @ dy.reshape(-1, cfg.d)
        do = mm(dy, p[f"Wo.{i}"].T).reshape(B, T, cfg.heads, cfg.dh).transpose(0, 2, 1, 3)
        dP = do @ vh.transpose(0, 1, 3, 2)
        dvh = P.transpose(0, 1, 3, 2) @ do
        dS = P * (dP - np.sum(dP * P, axis=-1, keepdims=True))
        dqh = (dS @ kh) * scale
        dkh = (dS.transpose(0, 1, 3, 2) @ qh) * scale
        dqr, dkr, dv = (a_.transpose(0, 2, 1, 3) for a_ in (dqh, dkh, dvh))
        dq = rope(dqr, cos, sin, inverse=True).reshape(B, T, cfg.d)
        dk = rope(dkr, cos, sin, inverse=True).reshape(B, T, cfg.d)
        dv = dv.reshape(B, T, cfg.d)
        hf = h.reshape(-1, cfg.d).T
        g[f"Wq.{i}"] += hf @ dq.reshape(-1, cfg.d)
        g[f"Wk.{i}"] += hf @ dk.reshape(-1, cfg.d)
        g[f"Wv.{i}"] += hf @ dv.reshape(-1, cfg.d)
        dh = mm(dq, p[f"Wq.{i}"].T) + mm(dk, p[f"Wk.{i}"].T) + mm(dv, p[f"Wv.{i}"].T)
        dxn, g[f"g1.{i}"] = rms_bwd(dh, c1)
        dx = dx + dxn

    np.add.at(g["E"], tokens, dx)
    return g


# ---------------------------------------------------------------- генерация

def generate(p, cfg, prompt_ids, max_new=200, temperature=0.0, top_k=0, seed=0, stop_ids=(EOS,)):
    rng = np.random.default_rng(seed)
    ids = list(prompt_ids)
    for _ in range(max_new):
        ctx = np.array(ids[-cfg.ctx:], dtype=np.int64)[None, :]
        logits, _ = forward(p, ctx, cfg, keep=False)
        l = logits[0, -1].astype(np.float64)
        if temperature <= 0:
            nxt = int(np.argmax(l))
        else:
            l = l / temperature
            if top_k:
                kth = np.sort(l)[-top_k]
                l = np.where(l < kth, -1e30, l)
            pr = np.exp(l - l.max())
            pr /= pr.sum()
            nxt = int(rng.choice(len(pr), p=pr))
        if nxt in stop_ids:
            break
        ids.append(nxt)
    return ids


# ---------------------------------------------------------------- токенизатор (байты)

def encode(text, bos=False):
    ids = [BOS] if bos else []
    for b in text.encode("utf-8"):
        ids.append(SPACE_ID if b == 0x20 else 3 + b)
    return ids


def decode(ids):
    out = bytearray()
    for t in ids:
        if t == SPACE_ID:
            out.append(0x20)
        elif 3 <= t <= 258:
            out.append(t - 3)
    return out.decode("utf-8", errors="replace")
