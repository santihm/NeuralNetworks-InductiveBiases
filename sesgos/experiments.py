"""Experimentos del proyecto. Cada función añade filas a results/raw/<exp>__<modelo>.csv y se salta
las ejecuciones ya hechas, así que se puede repartir el trabajo entre varias personas (cada una con
sus modelos) y relanzar sin repetir nada. El notebook llama a estas mismas funciones."""
import csv, glob, json, os, time
import numpy as np
import pandas as pd
import torch

from . import config as C
from .data import Data, nested_subsets
from .models import build, n_params
from .train import fit, memorize, set_seed, accuracy
from .transforms import pixel_permutation, block_permutation, apply_perm, shift, rotate, randomize_labels


class Store:
    """Resultados en CSV, uno por experimento y modelo, más checkpoints para H3 y la explicabilidad."""

    def __init__(self, root="results"):
        self.root = root
        os.makedirs(os.path.join(root, "raw"), exist_ok=True)
        os.makedirs(os.path.join(root, "checkpoints"), exist_ok=True)

    def path(self, exp, model):
        return os.path.join(self.root, "raw", f"{exp}__{model}__{C.PROTOCOL}.csv")

    def rows(self, exp, model=None):
        pattern = self.path(exp, model or "*")
        frames = [pd.read_csv(p) for p in sorted(glob.glob(pattern)) if os.path.getsize(p) > 0]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def done(self, exp, model, **key):
        df = self.rows(exp, model)
        if df.empty:
            return False
        mask = np.ones(len(df), dtype=bool)
        for k, v in key.items():
            if k not in df:
                return False
            mask &= (df[k].astype(str) == str(v)).to_numpy()
        return bool(mask.any())

    def add(self, exp, model, row):
        p = self.path(exp, model)
        new = not os.path.exists(p) or os.path.getsize(p) == 0
        with open(p, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)

    def add_many(self, exp, model, rows):
        for r in rows:
            self.add(exp, model, r)

    def claim(self, exp, model, **key):
        """Reserva una ejecución para este proceso (fichero de bloqueo creado de forma atómica).
        Devuelve False si otro proceso ya la tiene: así varios procesos se reparten el trabajo sin repetir."""
        d = os.path.join(self.root, "locks")
        os.makedirs(d, exist_ok=True)
        name = "__".join([exp, model] + [f"{k}={v}" for k, v in sorted(key.items())] + [C.PROTOCOL])
        try:
            fd = os.open(os.path.join(d, name + ".lock"), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()}\n".encode())
            os.close(fd)
            return True
        except FileExistsError:
            return False

    def skip(self, exp, model, **key):
        """True si la ejecución ya está hecha o la está haciendo otro proceso."""
        return self.done(exp, model, **key) or not self.claim(exp, model, **key)

    def ckpt(self, model, condition, seed):
        return os.path.join(self.root, "checkpoints", f"{model}__{condition}__s{seed}__{C.PROTOCOL}.pt")


def _meta(model_name, m, device):
    return {"n_params": n_params(m), "device": device, "torch": torch.__version__, "protocol": C.PROTOCOL,
            "time": time.strftime("%Y-%m-%d %H:%M:%S")}


def _perms():
    perm = pixel_permutation(C.PERM_SEED)
    bperm, order = block_permutation(C.PERM_SEED)
    return {"original": None, "pixeles": perm, "bloques": bperm}, order


def prep_fn(data, perm):
    return lambda x: data.norm(apply_perm(x, perm))


def chosen_lr(store, model):
    df = store.rows("lr", model)
    if df.empty:
        raise RuntimeError(f"Falta el barrido de learning rate de {model}: ejecuta lr_sweep primero.")
    return float(df.sort_values("best_val_acc", ascending=False).iloc[0]["lr"])


def lr_sweep(data, store, models=None, grid=None, seed=None, verbose=True):
    """Mismo barrido pequeño para los cuatro modelos, una semilla, elegido en validación."""
    models, grid = models or C.MODELS, grid or C.LR_GRID
    seed = C.LR_SWEEP_SEED if seed is None else seed
    xtr, ytr = data.train()
    xva, yva = data.val()
    prep = prep_fn(data, None)
    for name in models:
        for lr in grid:
            if store.skip("lr", name, lr=lr):
                continue
            set_seed(seed)
            m = build(name).to(data.device)
            r = fit(m, xtr, ytr, xva, yva, prep, lr=lr, seed=seed)
            store.add("lr", name, {"exp": "lr", "model": name, "lr": lr, "seed": seed, **r,
                                   **_meta(name, m, data.device)})
            if verbose:
                print(f"lr  {name:5s} lr={lr:.0e}  val={r['best_val_acc']:.4f}  ({r['seconds']:.0f} s)")


def baseline_h1(data, store, models=None, seeds=None, conditions=("original", "pixeles", "bloques"),
                verbose=True):
    """Baseline (condición original) y H1 (píxeles o bloques permutados con una permutación fija),
    con el train completo de 55 000 imágenes."""
    models, seeds = models or C.MODELS, C.SEEDS if seeds is None else seeds
    perms, _ = _perms()
    xtr, ytr = data.train()
    xva, yva = data.val()
    xte, yte = data.test()
    for name in models:
        lr = chosen_lr(store, name)
        for cond in conditions:
            prep = prep_fn(data, perms[cond])
            for seed in seeds:
                if store.skip("h1", name, condition=cond, seed=seed):
                    continue
                set_seed(seed)
                m = build(name).to(data.device)
                r = fit(m, xtr, ytr, xva, yva, prep, lr=lr, seed=seed)
                test = accuracy(m, xte, yte, prep)
                torch.save(m.state_dict(), store.ckpt(name, cond, seed))
                store.add("h1", name, {"exp": "h1", "model": name, "condition": cond, "seed": seed,
                                       "n_train": len(xtr), "lr": lr, **r, "test_acc": test,
                                       **_meta(name, m, data.device)})
                if verbose:
                    print(f"h1  {name:5s} {cond:8s} s{seed}  test={test:.4f}  ({r['seconds']:.0f} s)")


def h2(data, store, models=None, seeds=None, sizes=None, verbose=True):
    """Curvas de aprendizaje: subconjuntos estratificados y anidados del train; mismo val y test.
    El tamaño completo (55 000) se toma del baseline."""
    models, seeds, sizes = models or C.MODELS, C.SEEDS if seeds is None else seeds, sizes or C.H2_SIZES
    subsets = nested_subsets(data.y_all, data.train_idx, sizes, C.H2_SUBSET_SEED)
    xva, yva = data.val()
    xte, yte = data.test()
    prep = prep_fn(data, None)
    for name in models:
        lr = chosen_lr(store, name)
        for n in sizes:
            xtr, ytr = data.subset(subsets[n])
            for seed in seeds:
                if store.skip("h2", name, n_train=n, seed=seed):
                    continue
                set_seed(seed)
                m = build(name).to(data.device)
                r = fit(m, xtr, ytr, xva, yva, prep, lr=lr, seed=seed)
                test = accuracy(m, xte, yte, prep)
                store.add("h2", name, {"exp": "h2", "model": name, "n_train": n, "seed": seed, "lr": lr,
                                       **r, "test_acc": test, **_meta(name, m, data.device)})
                if verbose:
                    print(f"h2  {name:5s} n={n:5d} s{seed}  test={test:.4f}  ({r['seconds']:.0f} s)")


def h3(data, store, models=None, seeds=None, verbose=True):
    """Transformaciones solo en test sobre los modelos del baseline (sin reentrenar)."""
    models, seeds = models or C.MODELS, C.SEEDS if seeds is None else seeds
    xte, yte = data.test()
    prep = prep_fn(data, None)
    tests = ([("shift_h", k, lambda x, k=k: shift(x, dx=k)) for k in C.H3_SHIFTS] +
             [("shift_v", k, lambda x, k=k: shift(x, dy=k)) for k in C.H3_SHIFTS] +
             [("rotation", a, lambda x, a=a: rotate(x, a)) for a in C.H3_ANGLES + C.H3_EXTRA_ANGLES])
    for name in models:
        for seed in seeds:
            if store.skip("h3", name, seed=seed):
                continue
            path = store.ckpt(name, "original", seed)
            if not os.path.exists(path):
                raise RuntimeError(f"Falta {path}: ejecuta baseline_h1 de {name} antes que H3.")
            m = build(name).to(data.device)
            m.load_state_dict(torch.load(path, map_location=data.device))
            base = accuracy(m, xte, yte, prep)
            rows = []
            for kind, mag, f in tests:
                acc = accuracy(m, xte, yte, lambda x, f=f: prep(f(x)))
                rows.append({"exp": "h3", "model": name, "seed": seed, "kind": kind,
                             "magnitude": mag, "test_acc": acc, "retention": acc / base})
            store.add_many("h3", name, rows)
            if verbose:
                print(f"h3  {name:5s} s{seed}  base={base:.4f}")


def h4(data, store, models=None, seeds=None, fractions=None,
       weight_decays=None, max_epochs=None, verbose=True):
    """Memorización: 5 000 imágenes con una fracción de etiquetas aleatorizadas; lr del barrido,
    constante; con y sin weight decay. Guarda el resumen y la curva por época."""
    models, seeds = models or C.MODELS, C.SEEDS if seeds is None else seeds
    fractions = C.H4_FRACTIONS if fractions is None else fractions
    weight_decays = C.H4_WEIGHT_DECAYS if weight_decays is None else weight_decays
    max_epochs = C.H4_MAX_EPOCHS if max_epochs is None else max_epochs
    sub = nested_subsets(data.y_all, data.train_idx, [C.H4_N], C.H4_SUBSET_SEED)[C.H4_N]
    xtr, y_clean = data.subset(sub)
    xte, yte = data.test()
    prep = prep_fn(data, None)
    for name in models:
        lr = chosen_lr(store, name)
        for frac in fractions:
            y_noisy = torch.from_numpy(randomize_labels(data.y_all[sub], frac, C.H4_LABEL_SEED)).to(data.device)
            for wd in weight_decays:
                for seed in seeds:
                    if store.skip("h4", name, label_noise=frac, weight_decay=wd, seed=seed):
                        continue
                    set_seed(seed)
                    m = build(name).to(data.device)
                    curve, first = memorize(m, xtr, y_noisy, xte, yte, prep, lr=lr, seed=seed, wd=wd,
                                            max_epochs=max_epochs)
                    key = {"model": name, "label_noise": frac, "weight_decay": wd, "seed": seed}
                    store.add_many("h4curves", name, [{**key, **c} for c in curve])
                    last = curve[-1]
                    store.add("h4", name, {"exp": "h4", **key, "lr": lr, "n_train": len(sub),
                                           "epochs_run": last["epoch"],
                                           **{f"epoch_{int(t * 100)}": (e if e is not None else "")
                                              for t, e in first.items()},
                                           "final_train_acc": last["train_acc"],
                                           "final_test_acc": last["test_acc"],
                                           "best_test_acc": max(c["test_acc"] for c in curve if c["test_acc"] != ""),
                                           "seconds": last["seconds"], **_meta(name, m, data.device)})
                    if verbose:
                        print(f"h4  {name:5s} ruido={frac:.2f} wd={wd} s{seed}  "
                              f"train={last['train_acc']:.3f} ep95={first[C.H4_THRESHOLD]}  ({last['seconds']:.0f} s)")


def h4_control(data, store, controls=None, max_epochs=None, verbose=True):
    """Controles de H4: las mismas 5 000 imágenes y etiquetas, pero con otro learning rate (constante). Sirven para
    comprobar si el MLP que no memoriza y la CNN inestable con weight decay se deben al learning rate y no a la
    arquitectura. Resultados en results/raw/h4ctl y h4ctlcurves."""
    controls = C.H4_CONTROLS if controls is None else controls
    max_epochs = C.H4_MAX_EPOCHS if max_epochs is None else max_epochs
    sub = nested_subsets(data.y_all, data.train_idx, [C.H4_N], C.H4_SUBSET_SEED)[C.H4_N]
    xtr, _ = data.subset(sub)
    xte, yte = data.test()
    prep = prep_fn(data, None)
    for name, frac, wd, lr, seeds in controls:
        y_noisy = torch.from_numpy(randomize_labels(data.y_all[sub], frac, C.H4_LABEL_SEED)).to(data.device)
        for seed in seeds:
            if store.skip("h4ctl", name, label_noise=frac, weight_decay=wd, lr=lr, seed=seed):
                continue
            set_seed(seed)
            m = build(name).to(data.device)
            curve, first = memorize(m, xtr, y_noisy, xte, yte, prep, lr=lr, seed=seed, wd=wd, max_epochs=max_epochs,
                                    track_batch=True)
            key = {"model": name, "label_noise": frac, "weight_decay": wd, "lr": lr, "seed": seed}
            store.add_many("h4ctlcurves", name, [{**key, **c} for c in curve])
            last = curve[-1]
            store.add("h4ctl", name, {"exp": "h4ctl", **key, "n_train": len(sub), "epochs_run": last["epoch"],
                                      **{f"epoch_{int(t * 100)}": (e if e is not None else "") for t, e in first.items()},
                                      "final_train_acc": last["train_acc"], "final_test_acc": last["test_acc"],
                                      "seconds": last["seconds"], **_meta(name, m, data.device)})
            if verbose:
                print(f"h4ctl {name:5s} ruido={frac:.2f} wd={wd} lr={lr:g} s{seed}  train={last['train_acc']:.3f} "
                      f"ep95={first[C.H4_THRESHOLD]}  ({last['seconds']:.0f} s)")


def lr_extra(data, store, models=("mlp", "cnn", "lstm"), grid=None, verbose=True):
    """Un learning rate más allá de la rejilla, para los modelos que eligieron su extremo (3e-3). Se guarda aparte
    (results/raw/lrext) y no cambia el protocolo: solo dice si el extremo de la rejilla limitaba."""
    grid = C.LR_EXTRA if grid is None else grid
    xtr, ytr = data.train()
    xva, yva = data.val()
    prep = prep_fn(data, None)
    for name in models:
        for lr in grid:
            if store.skip("lrext", name, lr=lr):
                continue
            set_seed(C.LR_SWEEP_SEED)
            m = build(name).to(data.device)
            r = fit(m, xtr, ytr, xva, yva, prep, lr=lr, seed=C.LR_SWEEP_SEED)
            store.add("lrext", name, {"exp": "lrext", "model": name, "lr": lr, "seed": C.LR_SWEEP_SEED, **r,
                                      **_meta(name, m, data.device)})
            if verbose:
                print(f"lrext {name:5s} lr={lr:.0e}  val={r['best_val_acc']:.4f}  ({r['seconds']:.0f} s)")


def h2_resample(data, store, models=None, sizes=None, seeds=None, verbose=True):
    """H2 con pocos datos cambiando también el subconjunto en cada semilla (semilla del subconjunto = 31 + semilla),
    para que la desviación entre semillas incluya qué imágenes tocan. Resultados en results/raw/h2rs."""
    models = C.H2_RESAMPLE_MODELS if models is None else models
    sizes = C.H2_RESAMPLE_SIZES if sizes is None else sizes
    seeds = C.SEEDS if seeds is None else seeds
    xva, yva = data.val()
    xte, yte = data.test()
    prep = prep_fn(data, None)
    for name in models:
        lr = chosen_lr(store, name)
        for seed in seeds:
            subsets = nested_subsets(data.y_all, data.train_idx, sizes, C.H2_SUBSET_SEED + seed)
            for n in sizes:
                if store.skip("h2rs", name, n_train=n, seed=seed):
                    continue
                xtr, ytr = data.subset(subsets[n])
                set_seed(seed)
                m = build(name).to(data.device)
                r = fit(m, xtr, ytr, xva, yva, prep, lr=lr, seed=seed)
                test = accuracy(m, xte, yte, prep)
                store.add("h2rs", name, {"exp": "h2rs", "model": name, "n_train": n, "seed": seed,
                                         "subset_seed": C.H2_SUBSET_SEED + seed, "lr": lr, **r, "test_acc": test,
                                         **_meta(name, m, data.device)})
                if verbose:
                    print(f"h2rs {name:5s} n={n:5d} s{seed}  test={test:.4f}  ({r['seconds']:.0f} s)")


# --------------------------------------------------------------- controles exactos (baratos, CPU)
def mlp_equivalence(data, steps=600, seed=1):
    """Control de H1: un MLP entrenado con píxeles permutados y la primera capa inicializada con las
    columnas permutadas sigue exactamente la misma trayectoria que el MLP original. Se hace en float64
    porque en float32 el redondeo distinto de las sumas se amplifica durante el entrenamiento."""
    from .models import MLP
    perm = pixel_permutation(C.PERM_SEED)
    dev = "cpu"
    X, Y = data.train()
    Xv, Yv = data.val()
    X, Y, Xv, Yv = X.to(dev).double(), Y.to(dev), Xv.to(dev).double(), Yv.to(dev)
    norm = lambda x: (x - data.mean) / data.std
    set_seed(seed); a = MLP().double()
    set_seed(seed); b = MLP().double()
    with torch.no_grad():
        b.first_layer.weight.copy_(a.first_layer.weight[:, perm])
    ra = fit(a, X, Y, Xv, Yv, norm, lr=1e-3, seed=seed, steps=steps, eval_every=200, patience=0)
    rb = fit(b, X, Y, Xv, Yv, lambda x: norm(apply_perm(x, perm)), lr=1e-3, seed=seed, steps=steps,
             eval_every=200, patience=0)
    return {"val_original": ra["best_val_acc"], "val_permutado": rb["best_val_acc"],
            "max_diff_primera_capa": (a.first_layer.weight[:, perm] - b.first_layer.weight).abs().max().item(),
            "max_diff_resto": max((pa - pb).abs().max().item()
                                  for pa, pb in zip(list(a.parameters())[2:], list(b.parameters())[2:]))}


def vit_block_equivalence(seed=0, n=256):
    """Control de H1: permutar los bloques 4×4 equivale a permutar las posiciones aprendidas del ViT."""
    from .models import ViT
    bperm, order = block_permutation(C.PERM_SEED)
    perm = pixel_permutation(C.PERM_SEED)
    set_seed(seed); v = ViT().eval()
    w = ViT().eval(); w.load_state_dict(v.state_dict())
    with torch.no_grad():
        w.pos[0, 1:].copy_(v.pos[0, 1 + order])
    x = torch.rand(n, 1, 28, 28)
    with torch.no_grad():
        return {"diff_bloques": (v(x) - w(apply_perm(x, bperm))).abs().max().item(),
                "diff_pixeles": (v(x) - w(apply_perm(x, perm))).abs().max().item()}


# --------------------------------------------------------------- explicabilidad cuantitativa
def _load(store, name, cond, seed, device):
    path = store.ckpt(name, cond, seed)
    if not os.path.exists(path):
        return None
    m = build(name).to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    return m.eval()


def e1_smoothness(data, store, seeds=None, verbose=True):
    """E1: estructura espacial de la primera capa (correlación entre pesos vecinos) en la CNN (filtros 3×3)
    y el MLP (plantillas 28×28). En el MLP entrenado con píxeles permutados se mide antes y después de
    deshacer la permutación. Usa los checkpoints que haya; los resultados se guardan en results/raw/e1."""
    from .explain import neighbor_corr, first_layer_images
    seeds = C.SEEDS if seeds is None else seeds
    inv = torch.argsort(pixel_permutation(C.PERM_SEED))
    for name, conds in [("mlp", ["original", "pixeles"]), ("cnn", ["original", "pixeles", "bloques"])]:
        for cond in conds:
            for seed in seeds:
                if store.done("e1", name, condition=cond, seed=seed):
                    continue
                m = _load(store, name, cond, seed, "cpu")
                if m is None:
                    continue
                row = {"model": name, "condition": cond, "seed": seed,
                       "neighbor_corr": neighbor_corr(first_layer_images(m, name))}
                if name == "mlp" and cond == "pixeles":
                    row["neighbor_corr_despermutado"] = neighbor_corr(first_layer_images(m, name, inverse_perm=inv))
                store.add("e1", name, {"neighbor_corr_despermutado": "", **row})
                if verbose:
                    print("e1", row)


def e2_equivariance(data, store, seeds=None, shifts=range(1, 9), n=300, verbose=True):
    """E2: ¿se mueve la explicación con la prenda? Correlación entre el mapa de Integrated Gradients de la
    imagen desplazada y el mapa original desplazado, sobre las n primeras imágenes de test, para cada
    desplazamiento horizontal y vertical. Modelos del baseline (condición original)."""
    from .explain import ig_equivariance
    seeds = C.SEEDS if seeds is None else seeds
    x, y = data.Xte[:n].cpu(), data.Yte[:n].cpu()
    for name in C.MODELS:
        for seed in seeds:
            if store.done("e2", name, seed=seed):
                continue
            m = _load(store, name, "original", seed, "cpu")
            if m is None:
                continue
            rows = []
            for vertical in [False, True]:
                for k in shifts:
                    r = ig_equivariance(m, data.norm, x, y, k=k, vertical=vertical)
                    rows.append({"model": name, "seed": seed, "direction": "vertical" if vertical else "horizontal",
                                 "shift": k, "corr": float(r.mean()), "n": n})
            store.add_many("e2", name, rows)
            if verbose:
                print("e2", name, seed, [round(r["corr"], 2) for r in rows if r["direction"] == "horizontal"])
