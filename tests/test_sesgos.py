"""Comprobaciones rápidas (CPU, segundos). Ejecutar con: python -m pytest -q"""
import numpy as np
import torch

from sesgos import config as C
from sesgos.models import build, n_params, patchify, MLP, ViT
from sesgos.transforms import pixel_permutation, block_permutation, apply_perm, shift, rotate, randomize_labels
from sesgos.data import nested_subsets, stratified_val
from sesgos.experiments import vit_block_equivalence


def test_param_counts_comparable():
    counts = {m: n_params(build(m)) for m in C.MODELS}
    assert counts == {"mlp": 101706, "cnn": 99122, "lstm": 101674, "vit": 105546}
    assert max(counts.values()) / min(counts.values()) < 1.07


def test_output_shapes():
    x = torch.rand(3, 1, 28, 28)
    for m in C.MODELS:
        assert build(m)(x).shape == (3, 10)


def test_pixel_permutation_is_bijection():
    p = pixel_permutation(C.PERM_SEED)
    assert sorted(p.tolist()) == list(range(784))


def test_block_permutation_keeps_patch_interiors():
    bperm, order = block_permutation(C.PERM_SEED)
    img = torch.arange(784.).reshape(1, 1, 28, 28)
    assert torch.equal(patchify(apply_perm(img, bperm))[0], patchify(img)[0][order])


def test_mlp_input_permutation_equals_weight_permutation():
    perm = pixel_permutation(C.PERM_SEED)
    torch.manual_seed(0)
    a = MLP().double().eval()
    b = MLP().double().eval()
    b.load_state_dict(a.state_dict())
    with torch.no_grad():
        b.first_layer.weight.copy_(a.first_layer.weight[:, perm])
    x = torch.rand(64, 1, 28, 28).double()
    assert (a(x) - b(apply_perm(x, perm))).abs().max() < 1e-12


def test_vit_block_permutation_equals_position_permutation():
    r = vit_block_equivalence()
    assert r["diff_bloques"] < 1e-5 and r["diff_pixeles"] > 1e-2


def test_shift_and_rotate():
    x = torch.zeros(1, 1, 28, 28); x[..., 10, 10] = 1
    assert shift(x, dx=3)[..., 10, 13] == 1 and shift(x, dy=-2)[..., 8, 10] == 1
    assert torch.allclose(rotate(x, 0), x)
    y = torch.rand(4, 1, 28, 28)
    assert torch.allclose(rotate(rotate(y, 90), -90)[..., 4:24, 4:24], y[..., 4:24, 4:24], atol=1e-4)


def test_label_randomization_fraction():
    y = np.repeat(np.arange(10), 500)
    r = randomize_labels(y, 0.5, seed=1)
    changed = (r != y).mean()
    assert 0.42 < changed < 0.48          # 50 % sustituidas, ~10 % coinciden por azar


def test_splits_and_nested_subsets():
    y = np.repeat(np.arange(10), 600)
    tr, va = stratified_val(y, n_val=500, seed=0)
    assert len(np.intersect1d(tr, va)) == 0 and len(va) == 500
    s = nested_subsets(y, tr, [100, 300], seed=0)
    assert set(s[100]) <= set(s[300])
    assert np.all(np.bincount(y[s[100]]) == 10)
