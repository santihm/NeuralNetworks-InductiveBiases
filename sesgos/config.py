"""Protocolo experimental congelado. Cualquier cambio aquí afecta a los cuatro modelos por igual."""

MODELS = ["mlp", "cnn", "lstm", "vit"]
MODEL_LABELS = {"mlp": "MLP", "cnn": "CNN", "lstm": "LSTM", "vit": "Transformer"}

# Datos
N_VAL = 5000                 # validación sacada del train oficial (estratificada)
SPLIT_SEED = 2026            # semilla de la partición train/val
NORM_FROM = "train"          # media y desviación calculadas solo con train

# Entrenamiento común (baseline, H1, H2)
SEEDS = [0, 1, 2, 3, 4]
BATCH = 128
MAX_STEPS = 8000             # unas 19 épocas sobre 55 000 imágenes
WARMUP = 200
EVAL_EVERY = 250
WEIGHT_DECAY = 0.01          # AdamW por defecto, igual para todos
LR_GRID = [3e-4, 1e-3, 3e-3]
LR_SWEEP_SEED = 0
PATIENCE_EVALS = 0           # sin early stopping: todos entrenan MAX_STEPS y se queda el mejor checkpoint de validación
# Versión del protocolo. v1 usaba early stopping (8 evaluaciones sin mejora) y cortaba algunas ejecuciones de la
# CNN con entrada permutada en la fase ruidosa del principio; v2 entrena siempre los mismos pasos. Los resultados
# de cada versión van a ficheros distintos, así que nunca se mezclan.
PROTOCOL = "v2"

# Procesos en paralelo (Mac con MPS o GPU): tras el barrido de lr se lanzan trabajadores que se reparten las
# ejecuciones pendientes mediante ficheros de bloqueo. 0 = todo en un proceso.
PARALLEL_WORKERS = [["h4", "--models", "vit"], ["h4", "--models", "cnn,lstm,mlp"], ["h2"]]

# H1: permutaciones fijas (semilla propia, distinta de las de entrenamiento)
PERM_SEED = 777

# H2: tamaños de train (estratificados y anidados)
H2_SIZES = [100, 300, 1000, 3000, 10000]   # 55 000 es el baseline
H2_SUBSET_SEED = 31

# H3: transformaciones solo en test
H3_SHIFTS = [0, 1, 2, 3, 4, 5, 6]
H3_ANGLES = [0, 15, 30, 45, 60, 75, 90]
H3_EXTRA_ANGLES = [180]                    # se analiza aparte: hay prendas casi simétricas

# H4: memorización
H4_N = 5000
H4_SUBSET_SEED = 11
H4_LABEL_SEED = 12
H4_FRACTIONS = [0.0, 0.25, 0.5, 1.0]
H4_WEIGHT_DECAYS = [0.0, 0.1]              # sin y con regularización
H4_MAX_EPOCHS = 300
H4_THRESHOLD = 0.95                        # precisión de train que define "memorizado"
# Controles de H4 (añadidos tras ver los resultados, ver la sección 7 del notebook). Separan el efecto de la
# arquitectura del de la optimización. (modelo, fracción de etiquetas aleatorias, weight decay, learning rate, semillas)
H4_CONTROLS = [("mlp", 1.0, 0.0, 1e-3, [0, 1, 2, 3, 4]),   # ¿el MLP memoriza con un lr menor?
               ("mlp", 1.0, 0.0, 3e-4, [0, 1, 2, 3, 4]),
               ("mlp", 1.0, 0.0, 3e-3, [0, 1, 2, 3, 4]),   # el lr del protocolo, repetido en otro dispositivo
               ("lstm", 1.0, 0.0, 1e-3, [0, 1, 2]),        # mismo lr que el Transformer y el MLP de control
               ("cnn", 1.0, 0.0, 1e-3, [0, 1, 2]),
               ("cnn", 0.5, 0.0, 3e-3, [0, 1, 2])]         # misma condición que H4, midiendo también en modo train
# Comprobaciones de robustez (también añadidas tras los resultados): un learning rate por encima de la rejilla
# para los modelos que eligieron su extremo, y H2 con 100 y 300 imágenes cambiando el subconjunto en cada semilla.
LR_EXTRA = [1e-2]
H2_RESAMPLE_SIZES = [100, 300]
H2_RESAMPLE_MODELS = ["mlp", "cnn"]

# Estilo común de figuras y póster, con la paleta de las diapositivas del grupo (verde oscuro, verde azulado,
# ámbar y morado). Los tres modelos con sesgo pasan el validador de daltonismo (todas las parejas con ΔE >= 10
# en protanopía, deuteranopía y tritanopía); el MLP es la referencia sin sesgo y va en gris oscuro casi neutro,
# siempre con línea discontinua y marcador propio como codificación secundaria.
PALETTE = {"mlp": "#2F3B3A", "cnn": "#0F8A74", "lstm": "#C27612", "vit": "#6A4CA5"}
MARKERS = {"mlp": "o", "cnn": "s", "lstm": "^", "vit": "D"}
LINESTYLES = {"mlp": (0, (4, 2)), "cnn": "-", "lstm": "-", "vit": "-"}
INK = "#1F2A2A"
MUTED = "#5B6B6A"
GRID = "#E3E9E7"
DARK = "#10302F"        # fondo de la cabecera de las diapositivas y del póster
MINT = "#EAF2EF"        # paneles de apoyo


def smoke():
    """Modo de prueba rápida (CPU, minutos): mismo código, presupuesto mínimo. No produce resultados válidos."""
    g = globals()
    g.update(SEEDS=[0], MAX_STEPS=60, EVAL_EVERY=20, WARMUP=10, PATIENCE_EVALS=0, LR_GRID=[1e-3],
             H2_SIZES=[100, 1000], H4_N=500, H4_MAX_EPOCHS=3, H3_SHIFTS=[0, 2, 4], H3_ANGLES=[0, 45, 90])
