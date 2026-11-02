"""Estadística de los resultados: media ± σ, intervalo de confianza al 95 % y comparación de Welch.

Con 5 semillas las comparaciones tienen poca potencia; por eso se informa siempre la diferencia con su
intervalo, no solo el p-valor."""
import numpy as np
import pandas as pd
from scipy import stats as st

from . import config as C


def summary(df, by, col, scale=100.0):
    """Media, desviación típica, n e IC 95 % (t de Student) por grupo."""
    rows = []
    for key, g in df.groupby(by):
        x = pd.to_numeric(g[col], errors="coerce").dropna().to_numpy() * scale
        n = len(x)
        m = x.mean() if n else np.nan
        s = x.std(ddof=1) if n > 1 else np.nan
        h = st.t.ppf(0.975, n - 1) * s / np.sqrt(n) if n > 1 else np.nan
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(by, key)), "media": m, "sigma": s, "n": n, "ic95_inf": m - h, "ic95_sup": m + h})
    return pd.DataFrame(rows)


def _num(v, dec):
    return f"{v:.{dec}f}".replace(".", ",").replace("-", "−")


def pm_table(df, index, columns, col, scale=100.0, dec=1, index_name=None):
    """Tabla legible: «media ± σ» por celda, con los nombres de modelo en español y n semillas al pie."""
    g = df.groupby([index, columns])[col]
    m, s, n = g.mean() * scale, g.std() * scale, g.count()
    cell = {k: f"{_num(m[k], dec)} ± {_num(s[k], dec)}" if n[k] > 1 else _num(m[k], dec) for k in m.index}
    out = pd.Series(cell).unstack()
    rename = lambda v: C.MODEL_LABELS.get(v, v)
    out = out.rename(index=rename, columns=rename)
    order = [C.MODEL_LABELS[x] for x in C.MODELS]
    if set(out.columns) <= set(order):
        out = out[[c for c in order if c in out.columns]]
    if set(out.index) <= set(order):
        out = out.loc[[c for c in order if c in out.index]]
    out.index.name, out.columns.name = index_name or index, None
    return out


def welch(a, b):
    """Diferencia de medias (a - b), su IC 95 % de Welch y el p-valor bilateral."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return {"dif": a.mean() - b.mean() if len(a) and len(b) else np.nan,
                "ic95_inf": np.nan, "ic95_sup": np.nan, "p": np.nan}
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    d = a.mean() - b.mean()
    if se == 0:
        return {"dif": d, "ic95_inf": d, "ic95_sup": d, "p": 1.0 if d == 0 else 0.0}
    dof = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    h = st.t.ppf(0.975, dof) * se
    return {"dif": d, "ic95_inf": d - h, "ic95_sup": d + h, "p": 2 * st.t.sf(abs(d / se), dof)}


def _fmt(rows):
    """Tabla de comparaciones legible: diferencia en puntos, IC 95 % de Welch y p, con coma decimal."""
    out = pd.DataFrame(rows)
    out["diferencia (puntos)"] = out["dif"].map(lambda v: _num(v, 2))
    out["IC 95 %"] = [f"[{_num(a, 2)}, {_num(b, 2)}]" for a, b in zip(out.ic95_inf, out.ic95_sup)]
    out["p"] = out["p"].map(lambda p: "" if pd.isna(p) else (_num(p, 3) if p >= 0.001 else "< 0,001"))
    out = out.drop(columns=["dif", "ic95_inf", "ic95_sup"])
    return out[[c for c in out.columns if c != "p"] + ["p"]]


def h1_tests(h1):
    """Por modelo: caída de precisión (puntos) al permutar píxeles y bloques; y cada modelo frente al MLP."""
    rows = []
    for m in [x for x in C.MODELS if x in set(h1.model)]:
        g = h1[h1.model == m]
        base = g[g.condition == "original"].test_acc * 100
        for cond in ["bloques", "pixeles"]:
            r = welch(g[g.condition == cond].test_acc * 100, base)
            lab = {"bloques": "bloques permutados", "pixeles": "píxeles permutados"}[cond]
            rows.append({"comparación": f"{C.MODEL_LABELS[m]}: {lab} − original", **r})
    for cond in ["original", "pixeles"]:
        ref = h1[(h1.model == "mlp") & (h1.condition == cond)].test_acc * 100
        for m in [x for x in C.MODELS if x != "mlp" and x in set(h1.model)]:
            r = welch(h1[(h1.model == m) & (h1.condition == cond)].test_acc * 100, ref)
            rows.append({"comparación": f"{C.MODEL_LABELS[m]} − MLP ({'píxeles permutados' if cond == 'pixeles' else cond})", **r})
    return _fmt(rows)


def h2_advantage(h2, h1):
    """Ventaja de cada modelo sobre el MLP (puntos) en cada tamaño de train."""
    full = h1[h1.condition == "original"].assign(n_train=int(h1.n_train.max()))
    df = pd.concat([h2, full[h2.columns.intersection(full.columns)]], ignore_index=True)
    rows = []
    for n in sorted(df.n_train.unique()):
        ref = df[(df.model == "mlp") & (df.n_train == n)].test_acc * 100
        for m in [x for x in C.MODELS if x != "mlp" and x in set(df.model)]:
            r = welch(df[(df.model == m) & (df.n_train == n)].test_acc * 100, ref)
            rows.append({"imágenes de train": int(n), "modelo − MLP": C.MODEL_LABELS[m], **r})
    return _fmt(rows)


def h3_tests(h3, magnitude_shift=4, angle=45):
    """Retención (%) de cada modelo frente al MLP en una magnitud fija de cada transformación."""
    rows = []
    names = {"shift_h": f"horizontal {magnitude_shift} px", "shift_v": f"vertical {magnitude_shift} px",
             "rotation": f"rotación {angle}°"}
    for kind, mag in [("shift_h", magnitude_shift), ("shift_v", magnitude_shift), ("rotation", angle)]:
        d = h3[(h3["kind"] == kind) & (h3.magnitude == mag)]
        ref = d[d.model == "mlp"].retention * 100
        for m in [x for x in C.MODELS if x != "mlp" and x in set(d.model)]:
            r = welch(d[d.model == m].retention * 100, ref)
            rows.append({"transformación": names[kind], "modelo − MLP (retención)": C.MODEL_LABELS[m], **r})
    return _fmt(rows)


def h4_table(h4s, threshold="epoch_95"):
    """Épocas hasta el umbral de train por modelo, ruido y weight decay: media ± σ de las semillas que lo
    alcanzan, cuántas lo alcanzan, segundos hasta el umbral y precisión en test (al parar y pico).
    Los segundos son orientativos: se midieron en un Mac (MPS) con tres procesos en paralelo."""
    s = h4s.copy()
    s["ep"] = pd.to_numeric(s[threshold], errors="coerce")
    s["seg"] = s.seconds * s.ep / s.epochs_run          # tiempo hasta el umbral, suponiendo épocas de igual duración
    s["orden"] = s.model.map({m: i for i, m in enumerate(C.MODELS)})
    rows = []
    for (_, m, wd, f), g in s.groupby(["orden", "model", "weight_decay", "label_noise"]):
        ep = g.ep.dropna()
        rows.append({"modelo": C.MODEL_LABELS[m], "weight decay": _num(wd, 1), "etiquetas al azar": f"{f * 100:.0f} %",
                     "épocas hasta el umbral": "no llega" if ep.empty else
                     (_num(ep.mean(), 0) + (f" ± {_num(ep.std(), 0)}" if len(ep) > 1 else "")),
                     "semillas que llegan": f"{len(ep)}/{len(g)}",
                     "segundos hasta el umbral": "—" if ep.empty else _num(g.seg.mean(), 0),
                     "test al parar (%)": _num(g.final_test_acc.mean() * 100, 1),
                     "pico de test (%)": _num(g.best_test_acc.mean() * 100, 1)})
    return pd.DataFrame(rows)


def h4_control_table(ctl, h4s, threshold="epoch_95"):
    """Controles de H4 junto a la ejecución principal con la misma configuración (modelo, ruido, weight decay).
    Las filas del control indican el dispositivo: el protocolo se ejecutó en MPS y los controles en CPU."""
    main = h4s.assign(origen="protocolo")
    ctl = ctl.assign(origen="control")
    keys = ctl[["model", "label_noise", "weight_decay"]].drop_duplicates()
    df = pd.concat([main.merge(keys), ctl], ignore_index=True)
    df["ep"] = pd.to_numeric(df[threshold], errors="coerce")
    df["orden"] = df.model.map({m: i for i, m in enumerate(C.MODELS)})
    rows = []
    for (_, m, f, wd, lr, o), g in df.groupby(["orden", "model", "label_noise", "weight_decay", "lr", "origen"]):
        ep = g.ep.dropna()
        rows.append({"modelo": C.MODEL_LABELS[m], "etiquetas al azar": f"{f * 100:.0f} %", "weight decay": _num(wd, 1),
                     "learning rate": f"{lr:.0e}".replace("e-0", "e-"), "origen": f"{o} ({g.device.iloc[0]})",
                     "épocas hasta el umbral": "no llega" if ep.empty else
                     (_num(ep.mean(), 0) + (f" ± {_num(ep.std(), 0)}" if len(ep) > 1 else "")),
                     "semillas que llegan": f"{len(ep)}/{len(g)}",
                     "train al parar (%)": _num(g.final_train_acc.mean() * 100, 1),
                     "test al parar (%)": _num(g.final_test_acc.mean() * 100, 1)})
    return pd.DataFrame(rows)


def h2_resample_table(h2rs, h2):
    """H2 con pocos datos: subconjunto fijo (protocolo) frente a otro subconjunto por semilla (control).
    La diferencia CNN − MLP se calcula emparejada por semilla, porque ambos modelos ven el mismo subconjunto."""
    rows = []
    for origen, df in [("subconjunto fijo (protocolo)", h2), ("otro subconjunto por semilla (control)", h2rs)]:
        for n in sorted(set(h2rs.n_train)):
            p = df[df.n_train == n].pivot_table(index="seed", columns="model", values="test_acc") * 100
            if not {"mlp", "cnn"} <= set(p.columns):
                continue
            d = (p.cnn - p.mlp).dropna()
            h = st.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
            pv = 2 * st.t.sf(abs(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))), len(d) - 1)
            rows.append({"imágenes": n, "origen": origen,
                         "MLP (%)": f"{_num(p.mlp.mean(), 1)} ± {_num(p.mlp.std(), 1)}",
                         "CNN (%)": f"{_num(p.cnn.mean(), 1)} ± {_num(p.cnn.std(), 1)}",
                         "CNN − MLP (puntos)": f"{_num(d.mean(), 2)} ± {_num(d.std(ddof=1), 2)}",
                         "IC 95 % (emparejado)": f"[{_num(d.mean() - h, 2)}, {_num(d.mean() + h, 2)}]",
                         "p": _num(pv, 3) if pv >= 0.001 else "< 0,001",
                         "semillas en que gana la CNN": f"{int((d > 0).sum())}/{len(d)}"})
    return pd.DataFrame(rows)
