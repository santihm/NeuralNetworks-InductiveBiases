#!/bin/bash
# Lanza los trabajadores en paralelo (config.PARALLEL_WORKERS) junto a una ejecución ya en marcha.
# Se reparten las ejecuciones pendientes con ficheros de bloqueo, así que no se repite nada.
# Uso: nohup bash tools/workers_mac.sh < /dev/null >> results/logs/workers.out 2>&1 &
set -euo pipefail
exec < /dev/null
cd "$(dirname "$0")/.."
source .venv/bin/activate
python -u -m sesgos.run workers
