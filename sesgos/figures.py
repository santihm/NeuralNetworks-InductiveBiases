"""Figuras con el estilo común del proyecto (notebook y póster). Leen los CSV de results/raw."""
import glob, os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

from . import config as C

L = C.MODEL_LABELS
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def style(context="notebook"):
    """Registra Carlito (assets/fonts) si está y fija el estilo. context="poster" agranda la letra
    para que las figuras, colocadas a su tamaño final en el A0, midan lo mismo que el texto."""
    fams = []
    for f in glob.glob(os.path.join(ROOT, "assets", "fonts", "*.ttf")):
        font_manager.fontManager.addfont(f)
        fams.append(font_manager.FontProperties(fname=f).get_name())
    base = 10.5 if context == "notebook" else 28
    mpl.rcParams.update({
        # Carlito: la misma métrica que Calibri, la letra de las diapositivas del grupo (licencia OFL)
        "font.family": ["Carlito", "DejaVu Sans"] if "Carlito" in fams else ["DejaVu Sans"],
        "svg.fonttype": "none",
        "font.size": base, "axes.titlesize": base * 1.1, "axes.labelsize": base,
        "xtick.labelsize": base * 0.9, "ytick.labelsize": base * 0.9, "legend.fontsize": base * 0.9,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": base * 0.8,
        "text.color": C.INK, "axes.labelcolor": C.INK, "xtick.color": C.MUTED, "ytick.color": C.MUTED,
        "axes.edgecolor": C.MUTED, "axes.linewidth": 0.8 if context == "notebook" else 1.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": C.GRID,
        "grid.linewidth": 0.8 if context == "notebook" else 1.6,
        "lines.linewidth": 2 if context == "notebook" else 4,
        "lines.markersize": 6 if context == "notebook" else 13,
        "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 200,
        "savefig.bbox": "tight", "figure.facecolor": "white", "axes.facecolor": "white",
        "pdf.fonttype": 42,
    })
    return context


def save(fig, name, outdir="figures", svg=False):
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, f"{name}.pdf"))
    fig.savefig(os.path.join(outdir, f"{name}.png"))
    if svg:                                   # el póster (HTML) incrusta las figuras en SVG, con el texto como texto
        fig.savefig(os.path.join(outdir, f"{name}.svg"), transparent=True)


def _agg(df, by, col):
    g = df.groupby(by)[col]
    return g.mean().rename("mean").to_frame().join(g.std(ddof=1).rename("std")).join(g.size().rename("n")).reset_index()


def _seeds_note(df):
    n = _n_seeds(df)
    return f"media ± σ, {n} semilla" + ("s" if n != 1 else "")


def _note_y():
    return -0.22 if mpl.rcParams["font.size"] < 20 else -0.42


def _poster():
    return mpl.rcParams["font.size"] >= 20


def legend_top(fig, ax=None):
    """Leyenda de modelos en una fila, encima de los paneles (línea + marcador de cada modelo).
    En el póster no se dibuja: la clave de colores está una vez, debajo del mensaje principal."""
    if _poster():
        return
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=C.PALETTE[m], ls=C.LINESTYLES[m], marker=C.MARKERS[m], label=L[m]) for m in C.MODELS]
    if ax is not None:
        ax.legend(handles=h, loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=4, handlelength=2.6,
                  columnspacing=1.6, borderaxespad=0)
    else:
        fig.legend(handles=h, loc="upper left", bbox_to_anchor=(0.06, 0.99), ncol=4, handlelength=2.6,
                   columnspacing=1.6)


def present(df):
    """Modelos con datos, en el orden fijo del proyecto (el color sigue al modelo, nunca al orden)."""
    have = set(df["model"]) if "model" in df else set()
    return [m for m in C.MODELS if m in have]


def _n_seeds(df):
    return int(df.groupby([c for c in ["model", "condition", "n_train"] if c in df]).size().min())


def param_table(models):
    from .models import build, n_params
    rows = [{"Modelo": L[m], "Parámetros": n_params(build(m))} for m in models]
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- H1
def h1(df, ax=None):
    """Precisión en test por modelo: original, bloques 4x4 permutados y píxeles permutados."""
    conds = [("original", "Original", "o", "full"), ("bloques", "Bloques 4×4 permutados", "o", "left"),
             ("pixeles", "Píxeles permutados", "o", "none")]
    a = _agg(df, ["model", "condition"], "test_acc")
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(7.2, 3.4))
    ms = present(df)
    for i, m in enumerate(ms[::-1]):
        sub = a[a.model == m].set_index("condition")
        xs = [sub.loc[c, "mean"] * 100 for c, *_ in conds if c in sub.index]
        ax.plot([min(xs), max(xs)], [i, i], color=C.GRID, lw=mpl.rcParams["lines.linewidth"] * 2.5,
                solid_capstyle="round", zorder=1)
        for k, (c, lab, mk, fill) in enumerate(conds):
            if c not in sub.index:
                continue
            r = sub.loc[c]
            ax.errorbar(r["mean"] * 100, i + (k - 1) * 0.12, xerr=r["std"] * 100, fmt=mk,
                        color=C.PALETTE[m], mfc=C.PALETTE[m] if fill == "full" else "white",
                        fillstyle=fill, capsize=3, zorder=3,
                        label=lab if m == ms[0] else None)
    ax.set_yticks(range(len(ms)), [L[m] for m in ms[::-1]])
    ax.grid(axis="y", visible=False); ax.grid(axis="x", visible=True)
    ax.set_xlabel("Precisión en test (%)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, handletextpad=0.3, columnspacing=1.2)
    if not _poster():
        ax.text(1.0, _note_y(), _seeds_note(df), transform=ax.transAxes,
                ha="right", color=C.MUTED, fontsize=mpl.rcParams["font.size"] * 0.8)
    return fig


# ------------------------------------------------------------------- H2
def h2(df_h2, df_h1, axes=None):
    """Izquierda: precisión en test frente al tamaño del train (escala log, banda ±σ).
    Derecha: diferencia con el MLP en puntos, emparejando semillas (misma semilla, mismo subconjunto)."""
    full = df_h1[df_h1.condition == "original"].assign(n_train=int(df_h1.n_train.max()))
    df = pd.concat([df_h2, full[df_h2.columns.intersection(full.columns)]], ignore_index=True)
    a = _agg(df, ["model", "n_train"], "test_acc")
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
    fig = axes[0].figure
    ns = sorted(a.n_train.unique())
    ticks = [f"{int(n):,}".replace(",", " ") if n != 55000 else "55k" for n in ns]
    ticks = [t.replace(" 000", "k") if t.endswith(" 000") else t for t in ticks]
    ax = axes[0]
    for m in present(a):
        s = a[a.model == m].sort_values("n_train")
        x, y, e = s.n_train, s["mean"] * 100, s["std"].fillna(0) * 100
        ax.fill_between(x, y - e, y + e, color=C.PALETTE[m], alpha=0.15, lw=0)
        ax.plot(x, y, color=C.PALETTE[m], ls=C.LINESTYLES[m], marker=C.MARKERS[m], label=L[m])
    ax.set_ylabel("Precisión en test (%)")
    ax.set_title("Precisión")
    ax = axes[1]
    p = df.pivot_table(index=["n_train", "seed"], columns="model", values="test_acc")
    ax.axhline(0, color=C.PALETTE["mlp"], ls=C.LINESTYLES["mlp"], lw=mpl.rcParams["lines.linewidth"] * 0.6)
    ax.text(ns[0], 0.6, "MLP", color=C.PALETTE["mlp"], va="bottom", fontsize=mpl.rcParams["font.size"] * 0.85)
    if "mlp" in p:
        for m in [x for x in present(a) if x != "mlp"]:
            d = ((p[m] - p["mlp"]) * 100).groupby("n_train").agg(["mean", "std"]).dropna(subset=["mean"])
            ax.fill_between(d.index, d["mean"] - d["std"].fillna(0), d["mean"] + d["std"].fillna(0),
                            color=C.PALETTE[m], alpha=0.15, lw=0)
            ax.plot(d.index, d["mean"], color=C.PALETTE[m], ls=C.LINESTYLES[m], marker=C.MARKERS[m])
    ax.set_ylabel("Diferencia con el MLP (puntos)")
    ax.set_title("Ventaja del sesgo")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(ns, ticks)
        ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
        ax.set_xlabel("Imágenes de entrenamiento")
    fig.subplots_adjust(top=0.80 if mpl.rcParams["font.size"] < 20 else 0.88,
                        wspace=0.32 if mpl.rcParams["font.size"] < 20 else 0.42)
    legend_top(fig)
    if not _poster():
        axes[1].text(1.0, _note_y(), _seeds_note(df) + " (escala log en N)", transform=axes[1].transAxes,
                     ha="right", color=C.MUTED, fontsize=mpl.rcParams["font.size"] * 0.75)
    return fig


def _direct_labels(ax, a, xcol):
    last = a.sort_values(xcol).groupby("model").tail(1).sort_values("mean")
    ys = list(last["mean"] * 100)
    for i in range(1, len(ys)):                      # separa etiquetas que se pisan
        ys[i] = max(ys[i], ys[i - 1] + 1.2)
    for (_, r), y in zip(last.iterrows(), ys):
        ax.annotate(L[r.model], (r[xcol], r["mean"] * 100), xytext=(8, (y - r["mean"] * 100) * 4),
                    textcoords="offset points", va="center", color=C.INK,
                    fontsize=mpl.rcParams["font.size"] * 0.9)


def crossings(df_h2, df_h1, a="vit", b="mlp"):
    """Punto de cruce entre dos curvas por semilla (interpolación lineal en log N).
    Devuelve los cruces encontrados por semilla; vacío si las curvas no se cruzan."""
    full = df_h1[df_h1.condition == "original"].assign(n_train=int(df_h1.n_train.max()))
    df = pd.concat([df_h2, full[df_h2.columns.intersection(full.columns)]], ignore_index=True)
    out = []
    for seed, g in df.groupby("seed"):
        p = g.pivot_table(index="n_train", columns="model", values="test_acc").sort_index()
        if a not in p or b not in p:
            continue
        d = (p[a] - p[b]).to_numpy()
        n = np.log10(p.index.to_numpy(dtype=float))
        for i in range(len(d) - 1):
            if d[i] == 0 or d[i] * d[i + 1] < 0:
                t = d[i] / (d[i] - d[i + 1]) if d[i] != d[i + 1] else 0
                out.append({"seed": seed, "n_cruce": 10 ** (n[i] + t * (n[i + 1] - n[i]))})
    return pd.DataFrame(out)


# ------------------------------------------------------------------- H3
def h3(df, axes=None):
    """Retención (precisión transformada / precisión original) frente a la magnitud."""
    panels = [("shift_h", "Traslación horizontal", "píxeles"), ("shift_v", "Traslación vertical", "píxeles"),
              ("rotation", "Rotación", "grados")]
    if mpl.rcParams["font.size"] >= 20:
        panels = [(k, t.replace("Traslación ", "Desplaz. "), x) for k, t, x in panels]
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), sharey=True)
    fig = axes[0].figure
    for ax, (kind, title, xl) in zip(axes, panels):
        d = df[(df["kind"] == kind) & ~((kind == "rotation") & (df.magnitude > max(C.H3_ANGLES)))]
        a = _agg(d, ["model", "magnitude"], "retention")
        for m in present(a):
            s = a[a.model == m].sort_values("magnitude")
            x, y, e = s.magnitude, s["mean"] * 100, s["std"].fillna(0) * 100
            ax.fill_between(x, y - e, y + e, color=C.PALETTE[m], alpha=0.15, lw=0)
            ax.plot(x, y, color=C.PALETTE[m], ls=C.LINESTYLES[m], marker=C.MARKERS[m], label=L[m])
        ax.set_title(title)
        ax.set_xlabel(xl)
        ax.set_ylim(0, 105)
        ax.set_xticks([0, 30, 60, 90] if kind == "rotation" else [0, 2, 4, 6])
    axes[0].set_ylabel("Retención de precisión (%)")
    fig.subplots_adjust(top=0.80 if mpl.rcParams["font.size"] < 20 else 0.88)
    legend_top(fig)
    axes[-1].text(1.0, _note_y(), ("" if _poster() else "media ± σ entre semillas; ") + "retención = precisión transformada / original",
                  transform=axes[-1].transAxes, ha="right", color=C.MUTED, fontsize=mpl.rcParams["font.size"] * 0.8)
    return fig


# ------------------------------------------------------------------- H4
def h4(summary, curves, threshold="epoch_95", axes=None, ctl=None):
    """Izquierda: épocas hasta alcanzar el umbral de train frente a la fracción de ruido, con y sin
    weight decay (huecos = no alcanzado en el tope de épocas). Derecha: curvas con 50 % de ruido.
    `ctl` (resultados de h4ctl): añade el MLP con learning rate 10⁻³ y 100 % de etiquetas al azar."""
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
    fig = axes[0].figure
    ax = axes[0]
    s = summary.copy()
    cap = int(s.epochs_run.max()) if "epochs_run" in s else C.H4_MAX_EPOCHS
    s["ep"] = pd.to_numeric(s[threshold], errors="coerce")
    poster = _poster()
    # Media de las semillas que llegan al umbral (como en la tabla del notebook). Aro: alguna semilla no llega;
    # si no llega ninguna, el punto se dibuja en el tope de épocas.
    wds = [0.0] if poster else [0.0, max(C.H4_WEIGHT_DECAYS)]
    for wd in wds:
        d = s[s.weight_decay == wd]
        g = d.groupby(["model", "label_noise"])["ep"]
        a = pd.DataFrame({"mean": g.mean(), "std": g.std(), "frac": g.apply(lambda x: x.notna().mean())}).reset_index()
        a["mean"] = a["mean"].fillna(cap)
        for m in present(a):
            q = a[a.model == m].sort_values("label_noise")
            e = q["std"].fillna(0)
            yerr = [e, np.minimum(e, cap - q["mean"])]
            ax.errorbar(q.label_noise * 100, q["mean"], yerr=yerr, color=C.PALETTE[m],
                        ls=C.LINESTYLES[m] if wd == 0 else ":", marker=C.MARKERS[m], alpha=1.0 if wd == 0 else 0.45,
                        mfc="white" if wd else C.PALETTE[m], capsize=2, label=f"{L[m]}" if wd == 0 else None)
            hit = q[q.frac < 1]
            ax.scatter(hit.label_noise * 100, hit["mean"], s=(520 if poster else 120), facecolors="none",
                       edgecolors=C.INK, lw=(2.2 if poster else 1), zorder=4)
    ax.axhline(cap, color=C.MUTED, lw=0.8, ls="--")
    if ctl is not None and not ctl.empty:
        q = ctl[(ctl.model == "mlp") & (ctl.label_noise == 1.0) & (ctl.weight_decay == 0.0) & (np.isclose(ctl.lr, 1e-3))]
        ep = pd.to_numeric(q[threshold], errors="coerce").dropna()
        if len(ep):
            ax.errorbar([100], [ep.mean()], yerr=[[ep.std()], [ep.std()]], color=C.PALETTE["mlp"], marker="*",
                        ms=mpl.rcParams["lines.markersize"] * 2.0, ls="none", capsize=2, zorder=5)
            ax.annotate("estrella: MLP con lr 10⁻³", (100, ep.mean()), xytext=(97, 22), textcoords="data",
                        ha="right", va="center", color=C.PALETTE["mlp"], fontsize=mpl.rcParams["font.size"] * 0.8,
                        arrowprops=dict(arrowstyle="-", color=C.PALETTE["mlp"], lw=0.8, shrinkB=6))
    ax.set_xticks([f * 100 for f in C.H4_FRACTIONS])
    ax.set_xlabel("Etiquetas aleatorizadas (%)")
    ax.set_ylabel(f"Épocas hasta {threshold.split('_')[1]} % en train")
    ax.set_title("Ritmo de memorización")
    if poster:
        ax.annotate("aro: el MLP no llega con 3·10⁻³", (100, cap), xytext=(0, cap * 0.97), color=C.INK, va="top",
                    fontsize=mpl.rcParams["font.size"] * 0.85)
    else:
        ax.text(0.0, _note_y(), "Izquierda: continua sin weight decay, punteada con 0,1; media de las semillas que llegan; "
                "aro: alguna no llega en 300 épocas.\n"
                "Derecha: línea fina = train (etiquetas ruidosas), gruesa = test (etiquetas reales), banda = ±σ del test.",
                transform=ax.transAxes, color=C.MUTED, fontsize=mpl.rcParams["font.size"] * 0.8, va="top")
    ax = axes[1]
    c = curves[(curves.label_noise == 0.5) & (curves.weight_decay == 0.0)]
    for m in present(c):
        g = c[c.model == m].assign(test_acc=lambda d: pd.to_numeric(d.test_acc, errors="coerce"))
        # hasta la primera época en que alguna semilla para (tras memorizar): así la media es siempre de todas
        g = g[g.epoch <= g.groupby("seed").epoch.max().min()]
        sd = g.groupby("epoch")["test_acc"].std()
        g = g.groupby("epoch")[["train_acc", "test_acc"]].mean()
        if not poster:
            ax.plot(g.index, g.train_acc * 100, color=C.PALETTE[m], ls=C.LINESTYLES[m], lw=1.4)
        gt = g.test_acc.dropna()
        e = sd.reindex(gt.index).fillna(0)
        ax.fill_between(gt.index, (gt - e) * 100, (gt + e) * 100, color=C.PALETTE[m], alpha=0.15, lw=0)
        ax.plot(gt.index, gt * 100, color=C.PALETTE[m], ls=C.LINESTYLES[m], alpha=1.0 if poster else 0.55,
                lw=mpl.rcParams["lines.linewidth"] * 1.2, label=L[m])
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_xlabel("Época (escala log)")
    ax.set_ylabel("Precisión en test (%)" if poster else "Precisión (%)")
    ax.set_title("Test con 50 % de ruido" if poster else "Con 50 % de ruido")
    fig.subplots_adjust(top=0.80 if mpl.rcParams["font.size"] < 20 else 0.88,
                        wspace=0.32 if mpl.rcParams["font.size"] < 20 else 0.42)
    legend_top(fig)
    return fig


# ------------------------------------------------------------------- explicabilidad
def e2(df, axes=None):
    """E2: correlación entre la explicación (Integrated Gradients) de la imagen desplazada y la explicación
    original desplazada, frente al desplazamiento. 1 = la explicación se mueve con la prenda."""
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4), sharey=True)
    fig = axes[0].figure
    a = _agg(df, ["model", "direction", "shift"], "corr")
    for ax, d, t in zip(axes, ["horizontal", "vertical"], ["Horizontal", "Vertical"]):
        for m in present(a):
            q = a[(a.model == m) & (a.direction == d)].sort_values("shift")
            ax.fill_between(q["shift"], q["mean"] - q["std"].fillna(0), q["mean"] + q["std"].fillna(0),
                            color=C.PALETTE[m], alpha=0.15, lw=0)
            ax.plot(q["shift"], q["mean"], color=C.PALETTE[m], ls=C.LINESTYLES[m], marker=C.MARKERS[m])
        ax.axhline(0, color=C.MUTED, lw=0.8)
        ax.set_xticks(sorted(a["shift"].unique()))
        ax.set_xlabel("Desplazamiento (píxeles)")
        ax.set_title(t)
    axes[0].set_ylabel("¿Se mueve la explicación?")
    axes[0].set_ylim(-0.1, 1.0)
    fig.subplots_adjust(top=0.78 if mpl.rcParams["font.size"] < 20 else 0.88,
                        wspace=0.12 if mpl.rcParams["font.size"] < 20 else 0.16)
    legend_top(fig)
    if not _poster():
        axes[1].text(1.0, _note_y(), f"correlación de Integrated Gradients, media ± σ, {df.seed.nunique()} semillas",
                     transform=axes[1].transAxes, ha="right", color=C.MUTED, fontsize=mpl.rcParams["font.size"] * 0.75)
    return fig


def image_grid(imgs, rows, cols, ax_titles=None, cmap="RdBu_r", symmetric=True, figsize=None, title=None,
               row_labels=None, cbar_label=None):
    """Rejilla de imágenes, cada una con su propia escala (simétrica en torno a 0 si `symmetric`).
    `row_labels` rotula cada fila a la izquierda; `cbar_label` añade una barra de color común (escala relativa)."""
    fig, axes = plt.subplots(rows, cols, figsize=figsize or (cols * 0.9 + (0.9 if row_labels else 0),
                                                                 rows * 0.9 + 0.4), squeeze=False)
    im_ = None
    for k, ax in enumerate(axes.ravel()):
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.grid(False)
        if k < len(imgs):
            im = imgs[k]
            v = float(np.abs(im).max()) or 1.0
            im_ = ax.imshow(np.asarray(im) / v, cmap=cmap, vmin=-1 if symmetric else 0, vmax=1, interpolation="nearest")
            if ax_titles:
                ax.set_title(ax_titles[k], fontsize=mpl.rcParams["font.size"] * 0.7, loc="center")
    if row_labels:
        for r, lab in enumerate(row_labels):
            axes[r, 0].set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=mpl.rcParams["font.size"] * 0.8)
    if cbar_label and im_ is not None:
        cb = fig.colorbar(im_, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
        cb.set_label(cbar_label, fontsize=mpl.rcParams["font.size"] * 0.75)
        cb.ax.tick_params(labelsize=mpl.rcParams["font.size"] * 0.7)
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    return fig


QUESTIONS = {"H1": "¿Cae al\npermutar?", "H2": "¿Supera al MLP\ncon pocos datos?",
             "H3": "¿Tolera la\ntraslación?", "H4": "¿Memoriza antes\nque el MLP?"}


def verdict_matrix(pred, obs, ax=None):
    """Matriz modelos × hipótesis. pred/obs: dict[(modelo, "H1")] -> "si" | "parcial" | "no".
    Dentro de cada celda: izquierda = no, centro = en parte, derecha = sí.
    Círculo hueco = predicción; relleno = resultado."""
    hyps = list(QUESTIONS)
    x_of = {"no": -0.28, "parcial": 0.0, "si": 0.28}
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(7.4, 3.8))
    big, small = (420, 170) if mpl.rcParams["font.size"] < 20 else (2600, 1050)
    for j, m in enumerate(C.MODELS):
        for i, h in enumerate(hyps):
            for k in x_of.values():
                ax.plot(i + k, j, marker="|", color=C.GRID, ms=big ** 0.5 * 0.5, zorder=1)
            if (m, h) in pred:
                ax.scatter(i + x_of[pred[(m, h)]], j, s=big, facecolors="white", edgecolors=C.PALETTE[m],
                           lw=2, zorder=2)
            if (m, h) in obs:
                ax.scatter(i + x_of[obs[(m, h)]], j, s=small, color=C.PALETTE[m], zorder=3)
    ax.set_xticks(range(4), [f"{h}\n{q}" for h, q in QUESTIONS.items()])
    ax.xaxis.tick_top()
    ax.set_yticks(range(4), [L[m] for m in C.MODELS])
    ax.set_ylim(3.6, -0.6)                              # margen para que no se corten los círculos
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    for i in range(1, 4):
        ax.axvline(i - 0.5, color=C.GRID, lw=1)
    ax.set_xlim(-0.5, 3.5)
    ax.text(0, 1.0, "", transform=ax.transAxes)
    ax.set_xlabel("En cada celda: no · en parte · sí.   Hueco = predicción, relleno = resultado",
                  color=C.MUTED, fontsize=mpl.rcParams["font.size"] * 0.8)
    return fig


# Huecos de cada figura en el póster A0 (ancho y alto en mm). poster/build_poster.py usa los mismos valores.
POSTER_MM = {"h1": (380, 136), "h2": (380, 136), "h3": (380, 128), "h4": (380, 136), "e2_equivarianza": (300, 128),
             "veredictos": (330, 150)}


def all_figures(results="results", outdir="figures"):
    """Genera todas las figuras de resultados: tamaño notebook en figures/ y tamaño final del A0 en
    figures/poster/ (PDF, PNG y SVG), con la letra a la escala del póster."""
    from .experiments import Store
    from .verdicts import PRED, OBS
    st = Store(results)
    h1_, h2_, h3_, h4s_, h4c_, e2_, ctl_ = (st.rows(e) for e in ["h1", "h2", "h3", "h4", "h4curves", "e2", "h4ctl"])
    makers = [("h1", lambda: h1(h1_), not h1_.empty),
              ("h2", lambda: h2(h2_, h1_), not h2_.empty and not h1_.empty),
              ("h3", lambda: h3(h3_), not h3_.empty),
              ("h4", lambda: h4(h4s_, h4c_, ctl=ctl_), not h4s_.empty),
              ("e2_equivarianza", lambda: e2(e2_), not e2_.empty),
              ("veredictos", lambda: verdict_matrix(PRED, OBS), True)]
    mm = 1 / 25.4
    done = []
    for context, sub in [("notebook", ""), ("poster", "poster")]:
        style(context)
        for name, make, ok in makers:
            if not ok:
                continue
            fig = make()
            if context == "poster":
                w, h = POSTER_MM[name]
                fig.set_size_inches(w * mm, h * mm)
            save(fig, name, os.path.join(outdir, sub), svg=context == "poster")
            plt.close(fig)
            done.append(os.path.join(sub, name))
    style("notebook")
    return done
