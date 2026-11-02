"""Explicabilidad: filtros y pesos de la primera capa, mapas de saliencia y atención del ViT."""
import torch

from .models import ViT


def saliency(model, x, target=None):
    """|d logit_objetivo / d entrada| por píxel. x: (N,1,28,28) ya preparado para el modelo."""
    model.eval()
    x = x.clone().requires_grad_(True)
    out = model(x)
    target = out.argmax(1) if target is None else target
    out.gather(1, target.view(-1, 1)).sum().backward()
    return x.grad.abs().squeeze(1).detach()


def integrated_gradients(model, x, target, baseline=None, steps=32):
    """Integrated Gradients (Sundararajan et al., 2017) desde una referencia. Por defecto, ceros; con la entrada
    normalizada, la referencia natural es el fondo negro normalizado: baseline=norm(torch.zeros_like(x_crudo))."""
    model.eval()
    baseline = torch.zeros_like(x) if baseline is None else baseline
    total = torch.zeros_like(x)
    for a in torch.linspace(0, 1, steps, device=x.device):
        xi = (baseline + a * (x - baseline)).requires_grad_(True)
        out = model(xi).gather(1, target.view(-1, 1)).sum()
        total += torch.autograd.grad(out, xi)[0]
    return ((x - baseline) * total / steps).squeeze(1).detach()


def first_layer_images(model, name, inverse_perm=None):
    """Pesos de la primera capa como imágenes.
    CNN: los filtros 3×3. MLP: cada fila de la primera matriz como imagen 28×28; con `inverse_perm`
    se deshace la permutación de la entrada para comparar con el MLP original."""
    if name == "cnn":
        return model.first_layer.weight.detach().squeeze(1)            # (32, 3, 3)
    if name == "mlp":
        w = model.first_layer.weight.detach()                            # (112, 784)
        if inverse_perm is not None:
            w = w[:, inverse_perm]
        return w.reshape(-1, 28, 28)
    raise ValueError(name)


@torch.no_grad()
def vit_cls_attention(model: ViT, x):
    """Atención del token CLS sobre los 49 parches en cada bloque (media de cabezas), como (L, N, 7, 7)."""
    model.eval()
    t = model.tokens(x)
    maps = []
    for layer in model.encoder.layers:
        h = layer.norm1(t)
        a, w = layer.self_attn(h, h, h, need_weights=True, average_attn_weights=True)
        t = t + a
        t = t + layer._ff_block(layer.norm2(t))
        maps.append(w[:, 0, 1:].reshape(-1, 7, 7))
    return torch.stack(maps)


def neighbor_corr(w):
    """Correlación entre pesos vecinos (horizontal y vertical) de un conjunto de filtros o plantillas (K, H, W).
    Alta = estructura espacial suave; cerca de 0 = sin estructura espacial."""
    w = w.detach().float().cpu()
    a = torch.cat([w[:, :, :-1].reshape(-1), w[:, :-1, :].reshape(-1)])
    b = torch.cat([w[:, :, 1:].reshape(-1), w[:, 1:, :].reshape(-1)])
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def ig_equivariance(model, norm, x, y, k=4, steps=16, batch=100, vertical=False):
    """¿Se desplaza la explicación con la prenda? Correlación, imagen a imagen, entre el mapa de Integrated
    Gradients de la imagen desplazada k píxeles (a la derecha, o hacia abajo) y el mapa de la original desplazado k píxeles. 1 = la
    explicación se mueve exactamente con la prenda. x: imágenes sin normalizar (N,1,28,28); norm: normalización."""
    from .transforms import shift
    mv = (lambda t: shift(t, dy=k)) if vertical else (lambda t: shift(t, dx=k))
    out = []
    for i in range(0, len(x), batch):
        xb, yb = x[i:i + batch], y[i:i + batch]
        base = norm(torch.zeros_like(xb))                # referencia: imagen en negro (fondo)
        a = integrated_gradients(model, norm(xb), yb, base, steps).unsqueeze(1)
        b = integrated_gradients(model, norm(mv(xb)), yb, base, steps)
        a = mv(a).squeeze(1)                    # mueve el mapa original (relleno con 0 = sin atribución)
        a, b = a.reshape(len(xb), -1), b.reshape(len(xb), -1)
        a, b = a - a.mean(1, keepdim=True), b - b.mean(1, keepdim=True)
        out.append((a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-12))
    return torch.cat(out)
