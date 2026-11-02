"""Lanzador por línea de comandos para repartir los experimentos.

Ejemplos (desde la raíz del repo):
  python -m sesgos.run lr  --models cnn
  python -m sesgos.run h1  --models cnn --seeds 0-4
  python -m sesgos.run h2  --models lstm
  python -m sesgos.run h3
  python -m sesgos.run h4  --models mlp --seeds 0,1
  python -m sesgos.run all --smoke          # prueba completa en minutos (resultados no válidos)
  python -m sesgos.run extra                # controles de H4 y comprobaciones de robustez (tras el protocolo)
"""
import argparse, os, shutil, subprocess, sys

from . import config as C


def wait_for_lr(store, poll=30):
    import time
    while True:
        df = store.rows("lr")
        if not df.empty and all((df.model == m).sum() >= len(C.LR_GRID) for m in C.MODELS):
            return
        print("esperando al barrido de learning rate...", flush=True)
        time.sleep(poll)


def spawn_workers(results, smoke=False):
    """Lanza una sola vez los trabajadores en paralelo (no desde un trabajador, ni dos veces)."""
    if os.environ.get("SESGOS_WORKER") or not C.PARALLEL_WORKERS:
        return
    os.makedirs(os.path.join(results, "locks"), exist_ok=True)
    flag = os.path.join(results, "locks", f"workers_{C.PROTOCOL}.started")
    try:
        os.close(os.open(flag, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:
        return
    env = dict(os.environ, SESGOS_WORKER="1")
    keep_awake = ["caffeinate", "-dims"] if shutil.which("caffeinate") else []
    for i, args in enumerate(C.PARALLEL_WORKERS):
        os.makedirs(os.path.join(results, "logs"), exist_ok=True)
        log = open(os.path.join(results, "logs", f"worker_{i}.out"), "a")
        cmd = keep_awake + [sys.executable, "-u", "-m", "sesgos.run", *args, "--results", results] + (["--smoke"] if smoke else [])
        subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env,
                         start_new_session=True)
        print("trabajador", i, " ".join(args), flush=True)


def parse_seeds(s):
    if s is None:
        return None
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exp", choices=["lr", "h1", "h2", "h3", "h4", "all", "workers", "extra"])
    ap.add_argument("--models", default=None, help="lista separada por comas (por defecto, los cuatro)")
    ap.add_argument("--seeds", default=None, help="0-4 o 0,1,2")
    ap.add_argument("--data", default="data")
    ap.add_argument("--results", default="results")
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        C.smoke()
    if a.exp == "workers":                       # solo lanza los trabajadores en paralelo y termina
        spawn_workers(a.results, a.smoke)
        return
    from . import experiments as E
    from .data import Data
    from .train import pick_device
    dev = a.device or pick_device()
    data, store = Data(a.data, dev), E.Store(a.results)
    models = a.models.split(",") if a.models else None
    seeds = parse_seeds(a.seeds)
    print(f"dispositivo={dev} modelos={models or C.MODELS} semillas={seeds or C.SEEDS} smoke={a.smoke}")
    # Cada experimento ejecuta antes sus requisitos (lo ya hecho se salta): el barrido de lr va antes de todo
    # y H3 necesita los checkpoints del baseline (H1).
    needs = {"lr": ["lr"], "h1": ["lr", "h1"], "h2": ["lr", "h2"], "h3": ["lr", "h1", "h3"], "h4": ["lr", "h4"],
             "all": ["lr", "h1", "h3", "h2", "h4"], "extra": ["h4ctl", "h2rs", "lrext"]}
    steps = needs[a.exp]
    print(f"protocolo {C.PROTOCOL}: pasos {steps}")
    for s in steps:
        if s == "lr" and os.environ.get("SESGOS_WORKER"):
            wait_for_lr(store)                       # el barrido lo hace el proceso principal
            continue
        if s == "lr":
            E.lr_sweep(data, store, models)
            # los trabajadores en paralelo solo para el protocolo completo, no al pedir un modelo concreto
            if (not a.smoke or os.environ.get("SESGOS_TEST_WORKERS")) and models is None:
                spawn_workers(a.results, a.smoke)
        elif s == "h1":
            E.baseline_h1(data, store, models, seeds)
        elif s == "h2":
            E.h2(data, store, models, seeds)
        elif s == "h3":
            E.h3(data, store, models, seeds)
        elif s == "h4":
            E.h4(data, store, models, seeds)
        elif s == "h4ctl":
            E.h4_control(data, store)
        elif s == "h2rs":
            E.h2_resample(data, store)
        elif s == "lrext":
            E.lr_extra(data, store)


if __name__ == "__main__":
    main()
