"""Perturbaciones de la entrada (H1, H3) y de las etiquetas (H4)."""
import math
import numpy as np
import torch
import torch.nn.functional as F


def pixel_permutation(seed, n=784):
    """Permutación fija de los 784 píxeles."""
    return torch.from_numpy(np.random.default_rng(seed).permutation(n))


def block_permutation(seed, img=28, b=4):
    """Permutación que mueve los 49 bloques b×b enteros sin tocar su interior.
    Devuelve la permutación de píxeles y el orden de bloques (para el control exacto del ViT)."""
    nb = img // b
    order = np.random.default_rng(seed).permutation(nb * nb)
    perm = np.empty(img * img, dtype=np.int64)
    for dst, src in enumerate(order):
        sr, sc = divmod(int(src), nb)
        dr, dc = divmod(dst, nb)
        for i in range(b):
            for j in range(b):
                perm[(dr * b + i) * img + dc * b + j] = (sr * b + i) * img + sc * b + j
    return torch.from_numpy(perm), torch.from_numpy(order)


def apply_perm(x, perm):
    """x: (N,1,28,28). El píxel de salida p toma el valor del píxel de entrada perm[p]."""
    if perm is None:
        return x
    n = x.shape[0]
    return x.reshape(n, -1)[:, perm.to(x.device)].reshape(n, 1, 28, 28)


def shift(x, dx=0, dy=0):
    """Traslación entera; lo que entra es fondo (0). dx > 0 a la derecha, dy > 0 hacia abajo."""
    out = torch.zeros_like(x)
    H, W = x.shape[-2:]
    sy, ty = (slice(0, H - dy), slice(dy, H)) if dy >= 0 else (slice(-dy, H), slice(0, H + dy))
    sx, tx = (slice(0, W - dx), slice(dx, W)) if dx >= 0 else (slice(-dx, W), slice(0, W + dx))
    out[..., ty, tx] = x[..., sy, sx]
    return out


def rotate(x, degrees):
    """Rotación alrededor del centro con interpolación bilineal y fondo 0."""
    if degrees == 0:
        return x
    th = math.radians(degrees)
    A = torch.tensor([[math.cos(th), -math.sin(th), 0.0],
                      [math.sin(th), math.cos(th), 0.0]], device=x.device, dtype=x.dtype)
    grid = F.affine_grid(A.expand(x.shape[0], 2, 3), list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")


def randomize_labels(y, frac, seed, n_classes=10):
    """Sustituye una fracción de las etiquetas por una clase uniforme al azar.
    La etiqueta nueva coincide con la real un 10 % de las veces, como en Zhang et al. (2017)."""
    y = np.array(y, copy=True)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), int(round(frac * len(y))), replace=False)
    y[idx] = rng.integers(0, n_classes, len(idx))
    return y
