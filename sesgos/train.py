"""Bucles de entrenamiento comunes. Los cuatro modelos pasan exactamente por el mismo código."""
import math, random, time
import numpy as np
import torch
import torch.nn.functional as F

from . import config as C


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def accuracy(model, x, y, prep=None, bs=2000):
    model.eval()
    correct = 0
    for i in range(0, len(x), bs):
        xb = x[i:i + bs]
        if prep is not None:
            xb = prep(xb)
        correct += (model(xb).argmax(1) == y[i:i + bs]).sum().item()
    return correct / len(x)


def _schedule(step, total, warmup):
    return min(1.0, (step + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(step, total) / total))


def fit(model, xtr, ytr, xval, yval, prep, lr, seed, steps=None, bs=None,
        wd=None, warmup=None, eval_every=None, patience=None):
    """Entrena por pasos con AdamW y cosine decay, evalúa en validación cada `eval_every` pasos,
    se queda con el mejor checkpoint y para si no mejora en `patience` evaluaciones.

    `prep` convierte un lote de imágenes en [0,1] en la entrada del modelo (permutación + normalización).
    Devuelve un dict con la mejor precisión de validación, el paso en que se alcanzó y el tiempo."""
    steps = C.MAX_STEPS if steps is None else steps
    bs = C.BATCH if bs is None else bs
    wd = C.WEIGHT_DECAY if wd is None else wd
    warmup = C.WARMUP if warmup is None else warmup
    eval_every = C.EVAL_EVERY if eval_every is None else eval_every
    patience = C.PATIENCE_EVALS if patience is None else patience
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: _schedule(s, steps, warmup))
    n = len(xtr)
    order, pos = torch.randperm(n, generator=gen), 0
    best, best_step, best_state, stale = -1.0, 0, None, 0
    t0 = time.time()
    step = 0
    for step in range(1, steps + 1):
        if pos + bs > n:
            order, pos = torch.randperm(n, generator=gen), 0
        idx = order[pos:pos + min(bs, n)].to(xtr.device)
        pos += bs
        model.train()
        loss = F.cross_entropy(model(prep(xtr[idx])), ytr[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % eval_every == 0 or step == steps:
            acc = accuracy(model, xval, yval, prep)
            if acc > best:
                best, best_step, stale = acc, step, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
                if patience and stale >= patience:
                    break
    model.load_state_dict(best_state)
    return {"best_val_acc": best, "best_step": best_step, "steps_run": step,
            "seconds": time.time() - t0}


def memorize(model, xtr, ytr, xte, yte, prep, lr, seed, wd, max_epochs=None, bs=None, thresholds=None,
             track_batch=False):
    """H4: entrena por épocas con lr constante y sin early stopping. Registra por época la precisión
    en train (sobre las etiquetas usadas para entrenar, corruptas o no) y en test (etiquetas reales).
    El test se evalúa en todas las épocas hasta la 20 (donde está el pico de Arpit et al.) y después cada 5,
    para no multiplicar el coste; el train se evalúa en todas porque define el tiempo hasta memorizar.
    Devuelve la curva y la primera época en que se supera cada umbral (None si no se alcanza).
    Con track_batch=True registra además la precisión media de los lotes durante la época, en modo train
    (con las estadísticas del propio lote en BatchNorm): sirve para ver si la oscilación de la CNN viene de las
    estadísticas acumuladas de BatchNorm que usa la evaluación."""
    max_epochs = C.H4_MAX_EPOCHS if max_epochs is None else max_epochs
    bs = C.BATCH if bs is None else bs
    thresholds = (0.5, 0.8, C.H4_THRESHOLD) if thresholds is None else thresholds
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    curve, first = [], {t: None for t in thresholds}
    t0 = time.time()
    for ep in range(1, max_epochs + 1):
        model.train()
        hits = 0
        for idx in torch.randperm(len(xtr), generator=gen).split(bs):
            idx = idx.to(xtr.device)
            out = model(prep(xtr[idx]))
            loss = F.cross_entropy(out, ytr[idx])
            if track_batch:
                hits += (out.argmax(1) == ytr[idx]).sum().item()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        tr = accuracy(model, xtr, ytr, prep)
        stop = tr >= 0.999 or ep == max_epochs
        te = accuracy(model, xte, yte, prep) if (ep <= 20 or ep % 5 == 0 or stop) else None
        row = {"epoch": ep, "train_acc": tr, "test_acc": "" if te is None else te, "seconds": time.time() - t0}
        if track_batch:
            row["train_acc_batch"] = hits / len(xtr)
        curve.append(row)
        for t in thresholds:
            if first[t] is None and tr >= t:
                first[t] = ep
        if stop:
            break
    return curve, first
