#!/bin/bash
# Ejecuta el protocolo completo en un Mac con Apple Silicon (MPS), sin dejar que el Mac se duerma.
# Uso (desde la raíz del repo):
#   nohup bash tools/run_all_mac.sh < /dev/null >> results/logs/run_all.out 2>&1 &
# Es reanudable: si se corta, relanzar el mismo comando y se salta lo ya hecho.
set -euo pipefail
# Sin terminal: la entrada estándar puede quedar cerrada al cerrar Terminal y Python no arranca sin ella.
exec < /dev/null
cd "$(dirname "$0")/.."
mkdir -p results/logs

echo "== $(date '+%F %T') preparando entorno"
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install -q --upgrade pip
python -m pip install -q torch torchvision pandas scipy matplotlib nbformat nbclient ipykernel pytest
python - <<'EOF'
import torch, platform
print("python", platform.python_version(), "torch", torch.__version__, "mps", torch.backends.mps.is_available())
EOF
python -m pytest -q

echo "== $(date '+%F %T') experimentos"
# Por experimento y para los cuatro modelos, para tener pronto resultados de todos.
caffeinate -dims bash -c '
  set -e
  for exp in lr h1 h3 h2 h4; do
    echo "== $(date "+%F %T") $exp"
    python -u -m sesgos.run "$exp" < /dev/null
  done
'
echo "== $(date '+%F %T') FIN"
