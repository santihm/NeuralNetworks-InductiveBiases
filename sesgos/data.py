"""Fashion-MNIST, particiones fijas y las cuatro vistas de una misma imagen."""
import gzip, os
import numpy as np
import torch

from . import config as C

CLASSES = ["Camiseta", "Pantalón", "Jersey", "Vestido", "Abrigo",
           "Sandalia", "Camisa", "Zapatilla", "Bolso", "Botín"]
FILES = {"xtr": "train-images-idx3-ubyte.gz", "ytr": "train-labels-idx1-ubyte.gz",
         "xte": "t10k-images-idx3-ubyte.gz", "yte": "t10k-labels-idx1-ubyte.gz"}


def _read_idx(path):
    with gzip.open(path, "rb") as f:
        raw = f.read()
    ndim = raw[3]
    dims = [int.from_bytes(raw[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * ndim).reshape(dims).copy()


def _find_raw(root):
    for d in [root, os.path.join(root, "FashionMNIST", "raw")]:
        if all(os.path.exists(os.path.join(d, f)) for f in FILES.values()):
            return d
    return None


def load_raw(root="data"):
    """Descarga Fashion-MNIST con torchvision (https://github.com/zalandoresearch/fashion-mnist)
    si no está en `root`, y devuelve los arrays uint8 originales."""
    d = _find_raw(root)
    if d is None:
        from torchvision.datasets import FashionMNIST
        FashionMNIST(root, train=True, download=True)
        FashionMNIST(root, train=False, download=True)
        d = _find_raw(root)
    a = {k: _read_idx(os.path.join(d, f)) for k, f in FILES.items()}
    return a["xtr"], a["ytr"].astype(np.int64), a["xte"], a["yte"].astype(np.int64)


def stratified_val(y, n_val=C.N_VAL, seed=C.SPLIT_SEED):
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    val = np.concatenate([rng.choice(np.flatnonzero(y == c), n_val // len(classes), replace=False)
                          for c in classes])
    val = np.sort(val)
    return np.setdiff1d(np.arange(len(y)), val), val


def nested_subsets(y, idx, sizes, seed):
    """Subconjuntos estratificados y anidados de `idx`: el pequeño está contenido en el grande."""
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    order = {c: rng.permutation(idx[y[idx] == c]) for c in classes}
    return {n: np.sort(np.concatenate([order[c][: n // len(classes)] for c in classes])) for n in sizes}


class Data:
    """Contiene todo en el dispositivo como tensores (sin DataLoader: con modelos pequeños es más rápido).

    Las imágenes se guardan en [0, 1] y sin normalizar; `norm` se aplica después de cualquier
    transformación, para que el relleno de traslaciones y rotaciones sea el fondo real (0)."""

    def __init__(self, root="data", device="cpu"):
        xtr, ytr, xte, yte = load_raw(root)
        self.train_idx, self.val_idx = stratified_val(ytr)
        self.y_all = ytr
        to_x = lambda a: torch.from_numpy(a).float().div_(255).unsqueeze(1).to(device)
        to_y = lambda a: torch.from_numpy(a).to(device)
        self.X, self.Y = to_x(xtr), to_y(ytr)
        self.Xte, self.Yte = to_x(xte), to_y(yte)
        tr = self.X[torch.from_numpy(self.train_idx).to(device)]
        self.mean, self.std = tr.mean().item(), tr.std().item()
        self.device = device

    def norm(self, x):
        return (x - self.mean) / self.std

    def subset(self, idx):
        idx_t = torch.from_numpy(np.asarray(idx)).to(self.device)
        return self.X[idx_t], self.Y[idx_t]

    def train(self):
        return self.subset(self.train_idx)

    def val(self):
        return self.subset(self.val_idx)

    def test(self):
        return self.Xte, self.Yte
