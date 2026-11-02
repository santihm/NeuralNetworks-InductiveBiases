"""Construye el póster A0 (poster/poster.pdf) a partir de poster/poster.html y poster/poster.css.

1. Regenera las figuras de resultados a tamaño de póster (figures/poster/*.svg) desde la carpeta de resultados.
2. Dibuja los elementos gráficos propios del póster: la cuadrícula de la cabecera (una prenda de Fashion-MNIST
   ordenada y con sus píxeles permutados), cómo ve la imagen cada modelo y las plantillas del MLP (E1).
3. Sustituye cada marcador <!--svg:nombre--> del HTML por el SVG correspondiente e imprime el PDF con Chromium.

Uso (desde la raíz del repo):
    python tools/build_poster.py --results results
    python tools/build_poster.py --results results --no-figures   # solo maquetación (más rápido)
Requiere playwright con Chromium (pip install playwright && playwright install chromium).
"""
import argparse, os, re, sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sesgos import config as C                      # noqa: E402
from sesgos.data import Data                         # noqa: E402
from sesgos.transforms import pixel_permutation      # noqa: E402

POSTER = os.path.join(ROOT, "poster")
ASSETS = os.path.join(POSTER, "assets")
SAMPLE = 1                     # imagen de test que se dibuja (un jersey)


# ------------------------------------------------------------------ utilidades SVG
def _svg(w, h, body, cls=""):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:g} {h:g}" class="{cls}" '
            f'preserveAspectRatio="xMidYMid meet">{body}</svg>')


def _hex(c):
    c = c.lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))


def _mix(a, b, t):
    """Mezcla lineal entre dos colores hex (t=0 -> a, t=1 -> b)."""
    ra, rb = np.array(_hex(a)), np.array(_hex(b))
    r = (ra + (rb - ra) * float(t)).round().astype(int)
    return "#%02x%02x%02x" % tuple(r)


def _grid(img, x0, y0, cell, gap, on, off, thr=0.08):
    """Cuadrícula de celdas redondeadas: color `on` mezclado según la intensidad, `off` para el fondo."""
    out = []
    rows, cols = img.shape
    for i in range(rows):
        for j in range(cols):
            v = float(img[i, j])
            col = off if v < thr else _mix(off, on, 0.35 + 0.65 * min(1.0, v))
            out.append(f'<rect x="{x0 + j * (cell + gap):.2f}" y="{y0 + i * (cell + gap):.2f}" '
                       f'width="{cell}" height="{cell}" rx="{cell * 0.18:.2f}" fill="{col}"/>')
    return "".join(out)


def _small(img28, k=2):
    """Reduce 28x28 a (28/k)x(28/k) promediando bloques k x k y normaliza a [0, 1]."""
    a = img28.reshape(28 // k, k, 28 // k, k).mean(axis=(1, 3))
    return a / max(a.max(), 1e-6)


# ------------------------------------------------------------------ cabecera
def header_motif(img28):
    """La misma prenda, ordenada y con una permutación fija de sus celdas (la idea de H1)."""
    s = _small(img28)                                        # 14 x 14
    perm = np.random.default_rng(C.PERM_SEED).permutation(s.size)
    sp = s.ravel()[perm].reshape(s.shape)
    cell, gap = 10, 2.2
    side = 14 * cell + 13 * gap
    amber, off = "#E8A33D", "#1E4A48"
    w, h = side * 2 + 120, side + 40
    body = _grid(s, 0, 0, cell, gap, amber, off) + _grid(sp, side + 120, 0, cell, gap, amber, off)
    ax = side + 22
    body += (f'<path d="M{ax} {side / 2} H{ax + 70}" stroke="#E8A33D" stroke-width="4" fill="none"/>'
             f'<path d="M{ax + 62} {side / 2 - 10} L{ax + 76} {side / 2} L{ax + 62} {side / 2 + 10}" '
             f'stroke="#E8A33D" stroke-width="4" fill="none" stroke-linejoin="round"/>')
    lab = 'font-family="Carlito" font-size="15" fill="#9FC2BA" letter-spacing="1.5"'
    body += (f'<text x="0" y="{side + 30}" {lab}>ORIGINAL</text>'
             f'<text x="{side + 120}" y="{side + 30}" {lab}>PÍXELES PERMUTADOS</text>')
    return _svg(w, h, body, "motif")


# ------------------------------------------------------------------ cómo ve la imagen cada modelo
def model_views(img28):
    s = _small(img28)
    ink, off, cell, gap = "#1F2A2A", "#E3E9E7", 6, 1.2
    side = 14 * cell + 13 * gap
    views = {}

    # MLP: la imagen aplanada en un vector (aquí 196 celdas en 4 tiras de 49)
    flat = s.ravel()
    body = _grid(s, 0, 0, cell, gap, ink, off)
    x0 = side + 34
    body += f'<path d="M{side + 8} {side / 2} H{x0 - 8}" stroke="{C.PALETTE["mlp"]}" stroke-width="2.5"/>'
    c2 = 3.2
    for r in range(4):
        for j in range(49):
            v = flat[r * 49 + j]
            col = off if v < 0.08 else _mix(off, ink, 0.35 + 0.65 * v)
            body += (f'<rect x="{x0 + j * (c2 + 0.6):.2f}" y="{14 + r * 26:.2f}" width="{c2}" height="18" '
                     f'fill="{col}"/>')
    body += (f'<rect x="{x0 - 3}" y="8" width="{49 * (c2 + 0.6) + 5:.1f}" height="{3 * 26 + 30}" fill="none" '
             f'stroke="{C.PALETTE["mlp"]}" stroke-width="2" stroke-dasharray="6 4" rx="4"/>')
    views["mlp"] = _svg(x0 + 49 * (c2 + 0.6) + 6, side, body)

    # CNN: la misma ventana 3x3 se desliza por toda la imagen
    body = _grid(s, 0, 0, cell, gap, ink, off)
    col = C.PALETTE["cnn"]
    for (i, j, op) in [(3, 2, 0.35), (3, 5, 0.65), (3, 8, 1.0)]:
        body += (f'<rect x="{j * (cell + gap) - 1.5:.2f}" y="{i * (cell + gap) - 1.5:.2f}" '
                 f'width="{3 * cell + 2 * gap + 3:.2f}" height="{3 * cell + 2 * gap + 3:.2f}" rx="3" '
                 f'fill="{col}" fill-opacity="{0.18 * op:.2f}" stroke="{col}" stroke-opacity="{op}" stroke-width="2.6"/>')
    y = 3 * (cell + gap) + 3 * cell + 14
    body += (f'<path d="M{2 * (cell + gap)} {y} H{11 * (cell + gap) + 6}" stroke="{col}" stroke-width="2.5"/>'
             f'<path d="M{11 * (cell + gap)} {y - 6} L{11 * (cell + gap) + 8} {y} L{11 * (cell + gap)} {y + 6}" '
             f'stroke="{col}" stroke-width="2.5" fill="none"/>')
    views["cnn"] = _svg(side, side, body)

    # LSTM: lee las filas en orden, una por paso
    col = C.PALETTE["lstm"]
    body = ""
    for i in range(14):
        body += _grid(s[i:i + 1], 0, i * (cell + gap + 2.2), cell, gap, ink, off).replace("<rect", "<rect", 14)
    hh = 14 * (cell + gap + 2.2)
    body += (f'<path d="M{side + 14} 4 V{hh - 10}" stroke="{col}" stroke-width="2.6"/>'
             f'<path d="M{side + 8} {hh - 16} L{side + 14} {hh - 6} L{side + 20} {hh - 16}" stroke="{col}" '
             f'stroke-width="2.6" fill="none"/>')
    for i in (0, 4, 9, 13):
        yy = i * (cell + gap + 2.2) + cell / 2
        body += f'<circle cx="{side + 14}" cy="{yy:.1f}" r="3.2" fill="{col}"/>'
    views["lstm"] = _svg(side + 26, hh, body)

    # Transformer: 49 parches (aquí 7x7 bloques de 2x2) que se relacionan entre sí
    col = C.PALETTE["vit"]
    body, pg = "", 4.5
    centers = {}
    for bi in range(7):
        for bj in range(7):
            ox, oy = bj * (2 * cell + gap + pg), bi * (2 * cell + gap + pg)
            body += _grid(s[2 * bi:2 * bi + 2, 2 * bj:2 * bj + 2], ox, oy, cell, gap, ink, off)
            centers[(bi, bj)] = (ox + cell + gap / 2, oy + cell + gap / 2)
    a = centers[(3, 3)]
    for b in [(0, 1), (1, 5), (5, 1), (6, 5), (3, 6), (2, 0)]:
        q = centers[b]
        body = (f'<path d="M{a[0]:.1f} {a[1]:.1f} Q{(a[0] + q[0]) / 2 + 8:.1f} {(a[1] + q[1]) / 2 - 8:.1f} '
                f'{q[0]:.1f} {q[1]:.1f}" stroke="{col}" stroke-width="2.2" fill="none" stroke-opacity="0.85"/>') + body
    body += f'<circle cx="{a[0]:.1f}" cy="{a[1]:.1f}" r="5" fill="{col}"/>'
    side_v = 7 * (2 * cell + gap) + 6 * pg
    # las curvas van encima de la imagen para que se vean
    imgs = re.findall(r"<rect[^>]*/>", body)
    curves = re.findall(r"<path[^>]*/>|<circle[^>]*/>", body)
    views["vit"] = _svg(side_v, side_v, "".join(imgs) + "".join(curves))
    return views


# ------------------------------------------------------------------ E1: plantillas del MLP
def mlp_templates(results, n=6):
    """Primeras plantillas (filas de la primera capa) del MLP original y del entrenado con píxeles permutados,
    con la permutación deshecha. Si el MLP aprende lo mismo en otro orden, deben parecerse en estilo."""
    from sesgos.models import build
    out = []
    inv = torch.argsort(pixel_permutation(C.PERM_SEED))
    for cond in ["original", "pixeles"]:
        path = os.path.join(results, "checkpoints", f"mlp__{cond}__s0__{C.PROTOCOL}.pt")
        if not os.path.exists(path):
            return ""
        m = build("mlp")
        m.load_state_dict(torch.load(path, map_location="cpu"))
        w = m.first_layer.weight.detach()
        if cond == "pixeles":
            w = w[:, inv]
        out.append(w[:n].reshape(n, 28, 28).numpy())
    cell, gap, tile = 2.0, 0.0, 28 * 2.0
    neg, mid, pos = C.PALETTE["cnn"], "#F6F7F5", C.PALETTE["lstm"]
    body, sep = "", 10
    for r, ws in enumerate(out):
        for k, wk in enumerate(ws):
            v = float(np.abs(wk).max()) or 1.0
            ox, oy = k * (tile + sep), r * (tile + sep + 6)
            for i in range(28):
                for j in range(28):
                    t = float(wk[i, j]) / v
                    col = _mix(mid, pos, t) if t >= 0 else _mix(mid, neg, -t)
                    body += f'<rect x="{ox + j * cell:.1f}" y="{oy + i * cell:.1f}" width="{cell + 0.05}" height="{cell + 0.05}" fill="{col}"/>'
            body += f'<rect x="{ox}" y="{oy}" width="{tile}" height="{tile}" fill="none" stroke="#C9D3D0" stroke-width="0.8"/>'
    return _svg(n * (tile + sep) - sep, 2 * tile + sep + 6, body, "templates")


# ------------------------------------------------------------------ qué le hacemos a los datos
def _img_svg(a, x0, y0, size, ink="#1F2A2A", bg="#FFFFFF"):
    """Imagen 28x28 en escala de grises como cuadrícula de rectángulos (sin interpolar)."""
    c = size / 28
    out = [f'<rect x="{x0}" y="{y0}" width="{size}" height="{size}" fill="{bg}" stroke="#C9D3D0" stroke-width="0.8"/>']
    for i in range(28):
        for j in range(28):
            v = float(a[i, j])
            if v > 0.04:
                out.append(f'<rect x="{x0 + j * c:.2f}" y="{y0 + i * c:.2f}" width="{c + 0.05:.2f}" '
                           f'height="{c + 0.05:.2f}" fill="{_mix(bg, ink, min(1.0, v))}"/>')
    return "".join(out)


def manipulations(img):
    """La misma prenda con cada manipulación del estudio, para leer H1, H3 y H4 de un vistazo."""
    from sesgos.transforms import block_permutation, apply_perm, shift, rotate
    x = torch.tensor(img)[None, None]
    bperm, _ = block_permutation(C.PERM_SEED)
    items = [("Original", "", x.numpy()[0, 0]),
             ("Píxeles permutados", "H1", apply_perm(x, pixel_permutation(C.PERM_SEED)).numpy()[0, 0]),
             ("Bloques permutados", "H1", apply_perm(x, bperm).numpy()[0, 0]),
             ("Desplazada 4 px", "H3", shift(x, dx=4).numpy()[0, 0]),
             ("Girada 45°", "H3", rotate(x, 45).numpy()[0, 0]),
             ("Etiqueta al azar", "H4", img)]
    size, gap, top, out = 120, 26, 30, []
    txt = 'font-family="Carlito" font-size="16"'
    for k, (lab, h, t) in enumerate(items):
        x0 = k * (size + gap)
        if h:
            out.append(f'<rect x="{x0}" y="0" width="34" height="22" rx="4" fill="#10302F"/>'
                       f'<text x="{x0 + 17}" y="16" text-anchor="middle" {txt} font-weight="700" fill="#FFFFFF">{h}</text>')
        out.append(_img_svg(t, x0, top, size))
        out.append(f'<text x="{x0 + size / 2}" y="{top + size + 22}" text-anchor="middle" {txt} fill="#1F2A2A">{lab}</text>')
    x0 = 5 * (size + gap)
    out.append(f'<rect x="{x0 + 8}" y="{top + size - 30}" width="{size - 16}" height="24" rx="5" fill="#C27612"/>'
               f'<text x="{x0 + size / 2}" y="{top + size - 13}" text-anchor="middle" {txt} font-weight="700" '
               f'fill="#FFFFFF">«Bolso»</text>')
    return _svg(6 * size + 5 * gap, top + size + 30, "".join(out), "manip")


# ------------------------------------------------------------------ E2: mapas de Integrated Gradients
def ig_maps(results, data, models=None, k=4):
    """Integrated Gradients de la misma prenda sin desplazar (arriba) y desplazada k píxeles (abajo), semilla 0,
    con la correlación media de E2 para ese desplazamiento (5 semillas, 300 imágenes) debajo de cada modelo."""
    import glob
    import pandas as pd
    from sesgos.models import build
    from sesgos.explain import integrated_gradients
    from sesgos.transforms import shift
    from scipy.ndimage import gaussian_filter
    models = models or C.MODELS
    x = data.Xte[SAMPLE:SAMPLE + 1].cpu()
    y = data.Yte[SAMPLE:SAMPLE + 1].cpu()
    base = data.norm(torch.zeros_like(x))
    files = glob.glob(os.path.join(results, "raw", f"e2__*__{C.PROTOCOL}.csv"))
    e2 = pd.concat([pd.read_csv(f) for f in files]) if files else None
    size, gx, gy, top, left = 120, 22, 14, 30, 64
    neg, mid, pos = C.PALETTE["vit"], "#FFFFFF", C.PALETTE["lstm"]
    txt = 'font-family="Carlito"'
    body = ""
    for c, name in enumerate(models):
        path = os.path.join(results, "checkpoints", f"{name}__original__s0__{C.PROTOCOL}.pt")
        if not os.path.exists(path):
            return ""
        m = build(name)
        m.load_state_dict(torch.load(path, map_location="cpu"))
        m.eval()
        x0 = left + c * (size + gx)
        body += (f'<text x="{x0 + size / 2}" y="20" text-anchor="middle" {txt} font-size="19" font-weight="700" '
                 f'fill="{C.PALETTE[name]}">{C.MODEL_LABELS[name]}</text>')
        for r, d in enumerate([0, k]):
            a = integrated_gradients(m, data.norm(shift(x, dx=d)), y, base, 64)[0].numpy()
            a = gaussian_filter(a, 0.8)          # solo para verlo en el póster; r se calcula sin suavizar
            v = float(np.abs(a).max()) or 1.0
            y0 = top + r * (size + gy)
            cell = size / 28
            body += f'<rect x="{x0}" y="{y0}" width="{size}" height="{size}" fill="#FFFFFF" stroke="#C9D3D0" stroke-width="0.8"/>'
            for i in range(28):
                for j in range(28):
                    t = float(a[i, j]) / v
                    if abs(t) < 0.06:
                        continue
                    col = _mix(mid, pos, min(1, abs(t) * 1.6)) if t > 0 else _mix(mid, neg, min(1, abs(t) * 1.6))
                    body += (f'<rect x="{x0 + j * cell:.2f}" y="{y0 + i * cell:.2f}" width="{cell + 0.05:.2f}" '
                             f'height="{cell + 0.05:.2f}" fill="{col}"/>')
        if e2 is not None:
            q = e2[(e2.model == name) & (e2.direction == "horizontal") & (e2["shift"] == k)]["corr"]
            yy = top + 2 * size + gy + 30
            r = f"{q.mean():.2f}".replace(".", ",").replace("-", "−")
            body += (f'<text x="{x0 + size / 2}" y="{yy}" text-anchor="middle" {txt} font-size="26" font-weight="700" '
                     f'fill="{C.INK}">{r}</text>')
    lab = f'{txt} font-size="17" fill="{C.MUTED}" text-anchor="end"'
    body += (f'<text x="{left - 10}" y="{top + size / 2 + 6}" {lab}>original</text>'
             f'<text x="{left - 10}" y="{top + size + gy + size / 2 + 6}" {lab}>+{k} px</text>'
             f'<text x="{left - 10}" y="{top + 2 * size + gy + 30}" {lab}>r</text>')
    w = left + len(models) * (size + gx) - gx
    return _svg(w, top + 2 * size + gy + 40, body, "igmaps")


# ------------------------------------------------------------------ tabla de veredictos
def verdict_rows():
    """Filas HTML de la tabla predicción/resultado a partir de sesgos/verdicts.py."""
    from sesgos.verdicts import PRED, OBS
    sym = {"si": "✓", "parcial": "~", "no": "✗"}
    rows, n, ok = [], 0, 0
    for m in C.MODELS:
        cells = [f'<td>{C.MODEL_LABELS[m]}</td>']
        for h in ["H1", "H2", "H3", "H4"]:
            if (m, h) not in OBS:
                cells.append('<td><span class="pred">referencia</span></td>')
                continue
            o, p = OBS[(m, h)], PRED[(m, h)]
            n += 1
            ok += o == p
            miss = "" if o == p else " miss"
            cells.append(f'<td><span class="sym s-{o}">{sym[o]}</span>'
                         f'<span class="pred{miss}">previsto {sym[p]}</span></td>')
        rows.append(f'<tr class="{m}">' + "".join(cells) + "</tr>")
    return "".join(rows), f"{ok} de {n}"


# ------------------------------------------------------------------ montaje
def build(results, figures=True):
    if figures:
        from sesgos import figures as Fg
        Fg.all_figures(results, os.path.join(ROOT, "figures"))
    data = Data(os.path.join(ROOT, "data"), "cpu")
    img = data.Xte[SAMPLE, 0].numpy()
    parts = {"motif": header_motif(img), "templates": mlp_templates(results), "manip": manipulations(img),
             "igmaps": ig_maps(results, data)}
    for k, v in model_views(img).items():
        parts[f"view_{k}"] = v
    figdir = os.path.join(ROOT, "figures", "poster")
    for f in os.listdir(figdir):
        if f.endswith(".svg"):
            svg = open(os.path.join(figdir, f), encoding="utf-8").read()
            svg = svg[svg.index("<svg"):]                       # sin la cabecera XML
            svg = re.sub(r'width="[^"]*pt" height="[^"]*pt"', 'class="fig"', svg, count=1)
            parts[f[:-4]] = svg
    html = open(os.path.join(POSTER, "poster.html"), encoding="utf-8").read()
    missing = []

    def sub(m):
        k = m.group(1)
        if k not in parts:
            missing.append(k)
            return f'<div class="missing">falta {k}</div>'
        return parts[k]

    full = re.sub(r"<!--svg:([\w-]+)-->", sub, html)
    rows, hits = verdict_rows()
    full = full.replace("<!--verdict-rows-->", rows).replace("<!--verdict-hits-->", hits)
    out_html = os.path.join(POSTER, "build", "poster_full.html")
    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    full = full.replace('href="poster.css"', 'href="../poster.css"')
    open(out_html, "w", encoding="utf-8").write(full)
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page()
        pg.goto("file://" + out_html)
        pg.wait_for_timeout(500)
        over = pg.evaluate("() => { const p = document.querySelector('.page');"
                           " return [p.scrollHeight, p.clientHeight]; }")
        pg.pdf(path=os.path.join(POSTER, "poster.pdf"), width="841mm", height="1189mm",
               print_background=True, margin={"top": "0", "right": "0", "bottom": "0", "left": "0"})
        b.close()
    if missing:
        print("Faltan SVG:", missing)
    print("póster:", os.path.join(POSTER, "poster.pdf"), "| alto contenido/página (px):", over)
    return over


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()
    build(a.results, figures=not a.no_figures)
