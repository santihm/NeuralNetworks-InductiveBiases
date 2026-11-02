"""Genera sesgos_inductivos.ipynb a partir del código del paquete `sesgos`.

El notebook entregable es autocontenido: escribe cada módulo con %%writefile, así que funciona
sin el repo (Colab, otra máquina) y el código que se ve es exactamente el que se ejecuta.
Uso: python tools/build_notebook.py"""
import os
import nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULES = ["__init__", "config", "data", "transforms", "models", "train", "experiments", "explain", "figures", "verdicts", "stats"]

md = lambda s: nbf.v4.new_markdown_cell(s.strip("\n"))
code = lambda s: nbf.v4.new_code_cell(s.strip("\n"))

cells = []
cells += [md("""
# Sesgos inductivos: ¿cuándo importa la arquitectura?

**Redes de Neuronas · Máster Universitario en Inteligencia Artificial Aplicada · Universidad Carlos III de Madrid**

Alejandro Riego Velasco · Javier Aceituno Monja · Santiago Herron Mulet · Rodrigo Toldos Frutos

MLP, CNN, LSTM y Transformer incorporan supuestos distintos sobre la estructura de los datos: ninguno, localidad y pesos compartidos, orden secuencial, y relaciones entre todos los elementos. En este notebook comprobamos con experimentos cuándo esos sesgos ayudan, cuándo perjudican y por qué, sobre Fashion-MNIST, con cuatro modelos de unos 100 000 parámetros entrenados con el mismo protocolo y cinco semillas.

*Declaración de uso de IA: Durante la realización de esta actividad, se ha utilizado Claude (Anthropic) como apoyo en la programación, el análisis y la redacción.*
"""), md("""
## Cómo ejecutar este notebook

1. Copiar junto al notebook la carpeta `results/` que lo acompaña (CSV con cada ejecución y checkpoints de la semilla 0 a 4). Con ella, el modo por defecto, `"cargar"`, reproduce todas las tablas, figuras y explicaciones en unos minutos y sin entrenar.
2. Elegir el modo en la celda de configuración:
   - `"cargar"` (por defecto): no entrena; lee los CSV y checkpoints de `results/` y genera tablas y figuras.
   - `"completo"`: entrena todo lo que falte y guarda cada ejecución en `results/`. Si una ejecución ya está en los CSV, se salta, así que se puede interrumpir y relanzar. Desde cero son unos 360 entrenamientos: del orden de un día en un portátil con Apple Silicon (MPS) o en una GPU T4 de Colab.
   - `"prueba"`: el mismo código con un presupuesto mínimo, para comprobar en minutos que todo funciona. Sus números no valen como resultados.
3. *Reiniciar y ejecutar todo.* En Colab conviene *Entorno de ejecución → Cambiar tipo de entorno → GPU (T4)*; también funciona en CPU, más despacio.

**Dónde se obtuvieron los resultados.** Los CSV de `results/` se generaron con este mismo código en un Mac con Apple Silicon (MPS, torch 2.14.1). Con otro dispositivo o versión de PyTorch las operaciones en coma flotante no son idénticas, así que un reentrenamiento da resultados equivalentes en distribución (mismas medias dentro de su σ), no idénticos bit a bit.

**Datos.** Fashion-MNIST (Xiao, Rasul y Vollgraf, 2017), https://github.com/zalandoresearch/fashion-mnist. Se descarga automáticamente con `torchvision.datasets.FashionMNIST`.

**Reproducibilidad.** Semillas fijas para la partición (2026), las permutaciones (777), los subconjuntos (31 y 11), las etiquetas aleatorias (12) y el entrenamiento (0 a 4). Las librerías y versiones se imprimen en la celda siguiente; `requirements.txt` recoge las usadas.

**Mapa del código.** La sección 1 escribe el paquete `sesgos` con `%%writefile` (celdas plegadas: se despliegan con un clic). Qué hay en cada módulo:

| Módulo | Contenido |
|---|---|
| `config.py` | Protocolo común: semillas, partición, presupuesto, rejilla de learning rate, condiciones de cada hipótesis, colores |
| `data.py` | Descarga de Fashion-MNIST, partición estratificada, normalización |
| `transforms.py` | Permutaciones de píxeles y de bloques, traslación, rotación, etiquetas aleatorias |
| `models.py` | MLP, CNN, LSTM y Transformer, y recuento de parámetros |
| `train.py` | Entrenamiento por pasos (H1-H3) y por épocas hasta memorizar (H4), semillas, evaluación |
| `experiments.py` | Cada experimento: barrido de learning rate, H1-H4, controles, E1 y E2; guarda cada ejecución en CSV |
| `explain.py` | Integrated Gradients, atención del Transformer, pesos de la primera capa |
| `stats.py` | Media ± σ, intervalos de Welch y tablas |
| `figures.py` | Todas las figuras, del notebook y del póster |
| `verdicts.py` | Predicciones previas y veredictos observados, con sus criterios |
""")]

cells += [code("""
RUN_MODE = "cargar"          # "cargar" | "completo" | "prueba"
ONLY_MODELS = None           # p. ej. ["cnn"] para entrenar solo un modelo (reparto del cómputo)
ONLY_SEEDS = None            # p. ej. [0, 1, 2]
RESULTS = "results"          # carpeta de resultados (CSV + checkpoints)
DATA = "data"
USE_DRIVE = False            # en Colab: True para guardar los resultados en Google Drive
DRIVE_RESULTS = "/content/drive/MyDrive/RDN - Proyecto de Programación/results"

import os, sys, importlib, platform, warnings
warnings.filterwarnings("ignore", message=".*weights_only.*")      # aviso de torch.load; los checkpoints son nuestros
if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    RESULTS = DRIVE_RESULTS
import torch, torchvision, numpy as np, pandas as pd, matplotlib
import scipy
print(f"Python {platform.python_version()} · torch {torch.__version__} · torchvision {torchvision.__version__} · "
      f"numpy {np.__version__} · pandas {pd.__version__} · scipy {scipy.__version__} · matplotlib {matplotlib.__version__}")
""")]

cells += [md("""
## 1. Código común

Todo el proyecto usa un único paquete, `sesgos`. Las celdas siguientes escriben sus módulos en disco y después se importan. El protocolo completo está en `config.py`: cualquier cambio afecta igual a los cuatro modelos.
""")]
cells.append(code('import os\nos.makedirs("sesgos", exist_ok=True)'))
for name in MODULES:
    src = open(os.path.join(ROOT, "sesgos", f"{name}.py")).read().rstrip("\n")
    c = code(f"%%writefile sesgos/{name}.py\n{src}")
    c.metadata = {"jupyter": {"source_hidden": True, "outputs_hidden": True}, "collapsed": True}
    cells.append(c)

cells += [code("""
for m in [k for k in sys.modules if k == "sesgos" or k.startswith("sesgos.")]:
    del sys.modules[m]
from sesgos import config as C
if RUN_MODE == "prueba":
    C.smoke()
from sesgos import experiments as E, figures as Fg, explain as X, stats as T
from sesgos.data import Data, CLASSES
from sesgos.train import pick_device
from sesgos.models import build, n_params, patchify
from sesgos.transforms import pixel_permutation, block_permutation, apply_perm, shift
import matplotlib.pyplot as plt

DEVICE = pick_device()
TRAIN = RUN_MODE != "cargar"
data = Data(DATA, DEVICE)
store = E.Store(RESULTS)
Fg.style("notebook")
print(f"dispositivo: {DEVICE} · train {len(data.train_idx)} · val {len(data.val_idx)} · test {len(data.Yte)}")
""")]

cells += [md("""
## 2. Datos y las cuatro vistas de una misma imagen

Partición fija: 55 000 imágenes de entrenamiento y 5 000 de validación, estratificadas y sacadas del train oficial, y las 10 000 de test. La normalización usa la media y la desviación del train, y se aplica después de cualquier transformación, para que el relleno sea siempre el fondo real.

Cada modelo recibe la misma imagen y construye su vista: el MLP un vector de 784, la CNN la imagen 28×28, la LSTM una secuencia de 28 filas de 28 píxeles, y el Transformer una secuencia de 49 parches de 4×4.
"""), code("""
x0 = data.Xte[:1].cpu()
fig, ax = plt.subplots(1, 4, figsize=(10, 2.6))
ax[0].imshow(x0[0, 0], cmap="gray_r"); ax[0].set_title("CNN: imagen 28×28")
ax[1].plot(x0.reshape(-1).numpy(), color=C.PALETTE["mlp"], lw=0.6)
ax[1].set_title("MLP: vector de 784"); ax[1].set_xlabel("posición en el vector")
ax[2].imshow(x0[0, 0], cmap="gray_r")
for r in range(0, 28, 4): ax[2].axhline(r - 0.5, color=C.PALETTE["lstm"], lw=0.6)
ax[2].set_title("LSTM: 28 pasos (filas)")
ax[3].imshow(x0[0, 0], cmap="gray_r")
for r in range(0, 29, 4):
    ax[3].axhline(r - 0.5, color=C.PALETTE["vit"], lw=0.8); ax[3].axvline(r - 0.5, color=C.PALETTE["vit"], lw=0.8)
ax[3].set_title("Transformer: 49 parches 4×4")
for a in ax: a.set_xticks([]); a.set_yticks([]); a.grid(False)
ax[1].set_xticks([0, 392, 783]); ax[1].set_ylabel("intensidad")
plt.show()
""")]

cells += [md("""
## 3. Modelos con número de parámetros comparable

| Modelo | Configuración | Sesgo inductivo |
|---|---|---|
| MLP | 784 → 112 → 112 → 10, ReLU | Ninguno estructural |
| CNN | Conv 3×3 de 32, 64 y 136 canales con BatchNorm, max-pool tras las dos primeras, global average pooling | Localidad y pesos compartidos |
| LSTM | 1 capa, 144 unidades, lee la imagen fila a fila | Orden secuencial |
| Transformer | Parches 4×4 → 64 dimensiones, token CLS, posiciones aprendidas, 3 bloques, 4 cabezas, MLP de 128 | Relaciones entre todos los parches |

**Protocolo común:** AdamW (weight decay 0,01), batch 128, cosine decay con 200 pasos de warmup, como máximo 8 000 pasos, evaluación en validación cada 250 pasos, mejor checkpoint de validación y sin early stopping (todos los modelos entrenan los 8 000 pasos). El learning rate de cada modelo se elige con el mismo barrido {3·10⁻⁴, 10⁻³, 3·10⁻³} en validación. Tres modelos eligieron el extremo superior; un control posterior con 10⁻² (bajo la tabla del barrido) muestra que el MLP y la LSTM no mejorarían y la CNN sí, algo. Sin data augmentation, porque añadiría invariancias que contaminarían H1 y H3.
"""), code("""
Fg.param_table(C.MODELS)
"""), code("""
if TRAIN:
    E.lr_sweep(data, store, models=ONLY_MODELS)
lr = store.rows("lr")
display(T.pm_table(lr, "lr", "model", "best_val_acc", dec=2, index_name="learning rate · precisión en validación (%), semilla 0"))
print("elegido:", {C.MODEL_LABELS[m]: E.chosen_lr(store, m) for m in Fg.present(lr)})
lrx = store.rows("lrext")      # control posterior: un punto más allá de la rejilla (CPU); no cambia el protocolo
if len(lrx):
    display(T.pm_table(lrx, "lr", "model", "best_val_acc", dec=2,
                       index_name="control fuera de la rejilla · precisión en validación (%)"))
""")]

cells += [md("""
## 4. H1 · Estructura de la entrada

> «La ventaja de una arquitectura con sesgo inductivo desaparece si se destruye la estructura de la entrada.»

Aplicamos una permutación fija a todas las imágenes (train, validación y test) y reentrenamos los cuatro modelos. Usamos dos niveles:

- **Píxeles permutados:** se destruye la estructura local y la global.
- **Bloques 4×4 permutados:** se conserva cada bloque intacto y solo cambia dónde está. Separa la estructura local de la global.

**Predicción (escrita antes de los experimentos):** el MLP no cambia con ninguna de las dos, porque permutar la entrada equivale a permutar las columnas de su primera capa. La CNN cae con los píxeles permutados (pierde la localidad) y menos con los bloques. La LSTM cae porque cada paso deja de ser una fila. El Transformer no cambia con los bloques, porque sus posiciones aprendidas absorben esa permutación, y cae algo con los píxeles.
"""), code("""
if TRAIN:
    E.baseline_h1(data, store, models=ONLY_MODELS, seeds=ONLY_SEEDS)
h1 = store.rows("h1")
display(T.pm_table(h1, "condition", "model", "test_acc", dec=2, index_name="precisión en test (%), media ± σ, 5 semillas")
        .reindex(["original", "bloques", "pixeles"]))
fig = Fg.h1(h1); Fg.save(fig, "h1"); plt.show()
display(T.h1_tests(h1))     # diferencias en puntos de precisión, IC 95 % de Welch y p-valor
"""), md("""
**Controles exactos.** Dos comprobaciones de que las predicciones exactas no dependen del azar:

- Un MLP entrenado con píxeles permutados, con la primera capa inicializada con las columnas permutadas, sigue la misma trayectoria que el MLP original. Se hace en float64: en float32 el orden distinto de las sumas produce errores de redondeo que el entrenamiento amplifica.
- Permutar los bloques 4×4 equivale a permutar las posiciones aprendidas del Transformer; con los píxeles no hay equivalencia.
"""), code("""
print(E.vit_block_equivalence())
if RUN_MODE != "cargar":
    print(E.mlp_equivalence(data, steps=60 if RUN_MODE == "prueba" else 600))
"""), md("""
**Resultado: confirmada.** La ventaja de la CNN y de la LSTM desaparece al permutar los píxeles, el MLP no se ve afectado y el Transformer se comporta como se predijo: no nota los bloques y cae algo con los píxeles. Diferencias en puntos de precisión en test, con IC 95 % de Welch y 5 semillas (tabla anterior):

- **MLP.** +0,04 con bloques y +0,26 con píxeles. Lo esperado es exactamente cero: el control en float64 muestra que el MLP con la entrada permutada sigue la misma trayectoria que el original si se permuta igual su primera capa, y la inicialización aleatoria es i.i.d., así que ambos problemas son idénticos en distribución. El p = 0,05 en píxeles es compatible con ruido entre semillas: con 14 comparaciones en la tabla, con la corrección de Holm queda en p ≈ 0,15, no significativo, mientras que las caídas de la CNN, la LSTM y el Transformer con píxeles siguen siéndolo (p < 0,001).
- **CNN.** −3,9 con píxeles y −1,6 con bloques. No solo pierde la ventaja: pasa de superar al MLP por 2,7 puntos a quedar 1,5 por debajo. Sus filtros 3×3 y el pooling suponen que los píxeles vecinos están relacionados; con una permutación ese supuesto es falso y además le limita, porque una capa convolucional no puede combinar libremente píxeles lejanos como hace una capa densa. Con bloques conserva la estructura local dentro de cada bloque de 4×4 y recupera algo más de la mitad de la caída: lo que pierde es la relación entre bloques vecinos que agregan las capas 2 y 3.
- **LSTM.** −2,8 con píxeles y −1,9 con bloques. Cada paso es una fila: la permutación de píxeles convierte cada paso en 28 píxeles sueltos, y la de bloques reordena trozos de 4 píxeles dentro de cada fila y entre filas, así que también rompe la secuencia vertical que aprovecha.
- **Transformer.** Con bloques no cambia (+0,25, no significativo): permutar bloques equivale exactamente a permutar sus posiciones aprendidas (control anterior), y las aprende de nuevo. Con píxeles pierde 1,6: lo que destruye es la coherencia local dentro de cada parche, que es la única estructura espacial que el modelo recibe hecha. Con 55 000 imágenes y 100 000 parámetros, ese sesgo débil no le basta para superar al MLP (−0,9 puntos en el original).

La lectura conjunta: el sesgo ayuda en la medida en que el dato tiene la estructura que el sesgo supone, y cuando no la tiene puede costar más que no tener sesgo. La sección 8 lo muestra en los pesos de la primera capa.
""")]

cells += [md("""
## 5. H2 · Cantidad de datos

> «Cuantos menos datos, más importa el sesgo inductivo.»

Entrenamos con 100, 300, 1 000, 3 000, 10 000 y 55 000 imágenes (subconjuntos estratificados y anidados), con la misma validación y test. El presupuesto es el mismo en todos los tamaños: 8 000 pasos con el learning rate elegido para cada modelo, quedándonos con el mejor checkpoint de validación.

**Predicción:** la CNN por encima en todo el rango, con más ventaja relativa cuanto menos datos. El Transformer el peor con pocos datos, recortando distancia al crecer el conjunto; es improbable que alcance a la CNN con 55 000 imágenes. Los cruces esperables son del Transformer con el MLP y la LSTM.
"""), code("""
if TRAIN:
    E.h2(data, store, models=ONLY_MODELS, seeds=ONLY_SEEDS)
h2 = store.rows("h2")
fig = Fg.h2(h2, h1); Fg.save(fig, "h2"); plt.show()
full = h1[h1.condition == "original"].assign(n_train=int(h1.n_train.max()))
display(T.pm_table(pd.concat([h2, full[h2.columns.intersection(full.columns)]]), "n_train", "model", "test_acc",
                   index_name="imágenes de train · precisión en test (%), media ± σ"))
display(T.h2_advantage(h2, h1))   # ventaja sobre el MLP en cada tamaño (puntos, IC 95 % de Welch)
for a_, b_ in [("vit", "mlp"), ("vit", "lstm"), ("cnn", "vit"), ("lstm", "mlp")]:
    c = Fg.crossings(h2, h1, a_, b_)
    print(f"{a_} vs {b_}:", "no se cruzan" if c.empty else
          f"cruce en N ≈ {c.n_cruce.median():.0f} (rango {c.n_cruce.min():.0f}-{c.n_cruce.max():.0f}, {c.seed.nunique()} semillas)")
for a_ in ["cnn"]:
    c = Fg.crossings(h2, h1, a_, "mlp")
    print(f"{a_} vs mlp:", "no se cruzan" if c.empty else
          f"cruce en N ≈ {c.n_cruce.median():.0f} (rango {c.n_cruce.min():.0f}-{c.n_cruce.max():.0f}, {c.seed.nunique()} semillas)")
"""), md("""
**Resultado: confirmada, con un matiz importante.** Con pocos datos la arquitectura importa mucho más, pero no por la razón que esperábamos. Precisión media en test y separación entre el mejor y el peor modelo (las σ están en la primera tabla de arriba, entre 0,1 y 1,7 puntos):

| Imágenes | MLP | CNN | LSTM | Transformer | Mejor − peor |
|---|---|---|---|---|---|
| 100 | 68,6 | 67,3 | 62,4 | 50,1 | 18,6 |
| 300 | 75,7 | 76,5 | 72,3 | 62,1 | 14,4 |
| 1 000 | 80,5 | 81,7 | 79,9 | 73,3 | 8,5 |
| 3 000 | 83,0 | 85,1 | 83,8 | 79,9 | 5,2 |
| 10 000 | 85,7 | 88,3 | 86,9 | 83,8 | 4,5 |
| 55 000 | 89,3 | 92,0 | 90,5 | 88,4 | 3,6 |

- **La separación entre arquitecturas crece al bajar los datos**: 18,6 puntos entre el mejor y el peor modelo con 100 imágenes, frente a 3,6 con 55 000. En ese sentido, la hipótesis se cumple.
- **Pero no es el sesgo adecuado el que gana más con pocos datos, sino los sesgos débiles o poco adecuados los que pierden más.** La CNN no supera al MLP con 100 imágenes (−1,3 puntos, IC [−2,3, −0,3]; empata en dos semillas y pierde unos 2 puntos en tres) y su ventaja *crece* con los datos: +0,8 con 300 y +2,7 con 55 000. Con el subconjunto del protocolo, el cruce con el MLP está hacia las 230-250 imágenes en las tres semillas en que se produce; el control de abajo, cambiando el subconjunto en cada semilla, lo lleva más allá de las 300. La LSTM pierde 6,2 puntos con 100 imágenes y alcanza al MLP hacia las 1 500 (entre 1 300 y 1 900 según la semilla).
- **El Transformer es el que más depende de los datos**: −18,6 puntos con 100 imágenes, −7,2 con 1 000 y −0,9 con 55 000. La distancia se reduce en cada paso, pero no llega a cruzarse con ningún otro modelo. Es lo esperado de un modelo que tiene que aprender desde cero qué parches se relacionan entre sí (Dosovitskiy et al., 2021).

Por qué la CNN no gana con muy pocos datos: en Fashion-MNIST las prendas están centradas y escaladas, así que el MLP funciona como un comparador de plantillas fijas, y con 100 imágenes eso basta. La ventaja propia de la CNN, la tolerancia a desplazamientos (H3), no se necesita cuando todas las prendas están en el mismo sitio. Lo que sí aporta con más datos es combinar detectores locales que el MLP tiene que aprender píxel a píxel.

Respecto a la predicción escrita antes de los experimentos: acertamos que el Transformer sería el peor con pocos datos y que no alcanzaría a la CNN; fallamos en que la CNN estaría por encima en todo el rango con más ventaja cuanto menos datos, y en los cruces: el Transformer no se cruza con nadie, y los cruces reales son los de la CNN y la LSTM con el MLP.

**Límites.** El learning rate de cada modelo se eligió con 55 000 imágenes y se mantiene en todos los tamaños, y con 100 imágenes cada lote de 128 es el conjunto entero, así que 8 000 pasos son 8 000 épocas. Un ajuste por tamaño podría mover los puntos con pocos datos, aunque no parece probable que cierre una diferencia de 18 puntos. Además, el subconjunto de cada tamaño es el mismo en las cinco semillas (semilla 31): la σ refleja la variación de la inicialización y del orden de los lotes, no la de qué 100 imágenes tocan. Con 100 imágenes, la validación de 5 000 imágenes elige el checkpoint, lo que favorece por igual a los cuatro modelos.
"""), md("""
**Control: ¿depende de qué imágenes tocan?** Repetimos el MLP y la CNN con 100 y 300 imágenes, cambiando el subconjunto en cada semilla (semilla del subconjunto 31 + semilla; la semilla 0 coincide con el protocolo). Ambos modelos ven el mismo subconjunto en cada semilla, así que la diferencia se calcula emparejada. Se ejecutó en CPU.
"""), code("""
h2rs = store.rows("h2rs")
if len(h2rs):
    display(T.h2_resample_table(h2rs, h2))
else:
    print("Sin el control en results/raw (h2rs).")
"""), md("""
**Resultado del control.** Qué 100 imágenes tocan pesa tanto como la arquitectura: la precisión del MLP varía de 68,0 a 72,1 % según el subconjunto, una σ de 2,0 puntos frente a 0,6 con el subconjunto fijo. Aun así, la conclusión se mantiene y se refuerza: con 100 imágenes la CNN no gana al MLP en ninguna de las cinco semillas (−1,7 puntos emparejados, IC [−3,2, −0,2]). Con 300 imágenes, la ventaja de +0,8 del subconjunto fijo desaparece (−0,2, IC [−1,2, 0,7]): la CNN empieza a superar al MLP en algún punto entre 300 y 1 000 imágenes, no en 250. La tabla principal de H2 sigue siendo la del protocolo, con subconjunto fijo y anidado, que es la que permite comparar los seis tamaños.
""")]

cells += [md("""
## 6. H3 · Transformaciones solo en test

> «Una arquitectura generaliza a las transformaciones compatibles con su sesgo, pero no al resto.»

Sin reentrenar, evaluamos los modelos del baseline con dos transformaciones aplicadas solo en test. La métrica es la **retención**: precisión con la transformación dividida por la precisión sin ella.

- **Traslación (compatible con la CNN):** 0 a 6 píxeles, en horizontal y en vertical por separado. La convolución es equivariante a traslaciones y el global average pooling convierte esa equivarianza en casi invariancia.
- **Rotación (incompatible con todas):** 0 a 90°. Ninguna arquitectura tiene invariancia a la rotación incorporada.

| Modelo | Traslación horizontal | Traslación vertical | Rotación |
|---|---|---|---|
| MLP | No tolera | No tolera | No tolera |
| CNN | Tolera, sin ser perfecta | Tolera, sin ser perfecta | No tolera |
| LSTM | No tolera: cambia qué entrada ve cada unidad | En parte: equivale a retrasar la secuencia | No tolera |
| Transformer | No tolera: posiciones absolutas | No tolera | No tolera |

La CNN no es invariante del todo: el submuestreo del max-pooling rompe la equivarianza (Azulay y Weiss, 2019) y el relleno con ceros le permite aprender posiciones absolutas (Kayhan y van Gemert, 2020).
"""), code("""
if TRAIN:
    E.h3(data, store, models=ONLY_MODELS, seeds=ONLY_SEEDS)
h3 = store.rows("h3")
fig = Fg.h3(h3); Fg.save(fig, "h3"); plt.show()
sel = h3[((h3["kind"] != "rotation") & (h3.magnitude == 4)) | ((h3["kind"] == "rotation") & (h3.magnitude == 45))]
display(T.pm_table(sel, "kind", "model", "retention", index_name="retención (%), 4 px y 45°, media ± σ")
        .rename(index={"shift_h": "horizontal 4 px", "shift_v": "vertical 4 px", "rotation": "rotación 45°"}))
display(T.h3_tests(h3))           # retención frente al MLP con 4 píxeles y 45° (puntos, IC 95 % de Welch)
# Retención descontando el azar: (precisión − 0,1) / (precisión original − 0,1). 0 = azar, 1 = sin pérdida
sel = sel.assign(ret_azar=(sel.test_acc - 0.1) / (sel.test_acc / sel.retention - 0.1))
display(T.pm_table(sel, "kind", "model", "ret_azar", index_name="retención descontando el azar (%)")
        .rename(index={"shift_h": "horizontal 4 px", "shift_v": "vertical 4 px", "rotation": "rotación 45°"}))
display(T.pm_table(h3[h3["kind"] == "rotation"], "magnitude", "model", "retention", index_name="rotación (grados) · retención (%)"))
"""), md("""
**Resultado: confirmada, con dos matices.** Retención media con 4 píxeles de traslación y 45° de rotación (diferencias frente al MLP en la tabla anterior):

| Modelo | Horizontal 4 px | Vertical 4 px | Rotación 45° |
|---|---|---|---|
| MLP | 29,7 % | 60,1 % | 19,5 % |
| CNN | 94,7 % | 91,8 % | 27,3 % |
| LSTM | 33,7 % | 85,6 % | 22,8 % |
| Transformer | 46,5 % | 71,8 % | 28,7 % |

- **La CNN tolera la traslación en las dos direcciones** (+65 y +32 puntos sobre el MLP), porque la convolución es equivariante y el global average pooling resume la posición. No es invariante del todo: en horizontal, a 5 y 6 píxeles baja al 88 y al 78 % (en vertical, al 89 y al 85 %), cuando la prenda empieza a salirse del marco y pesan el submuestreo del pooling y el relleno con ceros, que permiten aprender posiciones absolutas (Azulay y Weiss, 2019; Kayhan y van Gemert, 2020).
- **La LSTM tolera la traslación vertical y no la horizontal**, tal como predice su sesgo: un desplazamiento vertical es un retraso de la secuencia de filas, que procesa con los mismos pesos en cada paso; uno horizontal cambia qué píxel entra en cada unidad, y ahí apenas mejora al MLP (+4 puntos, IC [1, 7]).
- **El Transformer tolera en parte** (+17 y +12 sobre el MLP), cuando habíamos previsto que no toleraría (tabla de predicciones de arriba). Aplica el mismo embedding a todos los parches, que es una forma débil de compartir pesos, pero sus posiciones son absolutas. Tampoco aprovecha que 4 píxeles muevan parches enteros: con 2 píxeles retiene el 72,7 % y con 4 el 46,5 %.
- **La traslación vertical perjudica menos a todos salvo a la CNN**, incluido el MLP (60 % frente a 30 % en horizontal), porque las prendas ocupan casi todo el alto de la imagen. La CNN retiene algo más en horizontal hasta 4 píxeles (94,7 frente a 91,8 %). Por eso cada modelo se compara con el MLP en la misma dirección.
- **Rotación: ninguna arquitectura la tolera.** A 45° todas retienen menos del 30 %; la CNN aguanta algo mejor los giros pequeños (84 % a 15°, frente al 59 % del MLP), que se parecen a deformaciones locales. A 180° la retención sube al 22-36 % porque bolsos, pantalones y vestidos son casi simétricos.

Descontando el azar (tercera tabla), las conclusiones no cambian: la CNN retiene el 94 y el 91 % con 4 píxeles, el MLP el 21 y el 55 %, y a 45° todos quedan por debajo del 20 %. Todos salvo la CNN retienen más en vertical, así que cada uno se compara con el MLP en la misma dirección.

La sección 8 (E2) lo comprueba en las explicaciones: en la CNN el mapa de Integrated Gradients se desplaza con la prenda, en la LSTM solo en vertical, y en el MLP y el Transformer no.
""")]

cells += [md("""
## 7. H4 · Memorización

> «Las cuatro arquitecturas pueden memorizar etiquetas aleatorias, pero no al mismo ritmo.»

Subconjunto estratificado de 5 000 imágenes. Sustituimos el 0, 25, 50 o 100 % de las etiquetas por una clase uniforme al azar, que coincide con la real un 10 % de las veces (como en Zhang et al., 2017). Las etiquetas aleatorias se sortean una vez (semilla 12) y son las mismas en las cinco semillas de entrenamiento. Entrenamos con el learning rate de cada modelo, constante, sin warmup y sin early stopping, hasta 300 épocas; solo paramos antes si el train llega al 99,9 %, es decir, cuando ya ha memorizado. Cada época medimos la precisión en train sobre las etiquetas usadas para entrenar; la de test, sobre las etiquetas reales, cada época hasta la 20 (donde se espera el pico) y después cada 5, para no multiplicar el coste. Por eso el test «final» de cada ejecución corresponde a la época en que paró.

**Tiempo hasta memorizar:** primera época en que el train supera el 50, el 80 y el 95 %, y los segundos hasta entonces. Si no se alcanza en 300 épocas, la ejecución se marca como no alcanzada. Los segundos son orientativos: se midieron en un Mac con tres entrenamientos en paralelo.

**Regularización:** weight decay de AdamW 0 frente a 0,1, igual para los cuatro modelos.

**Predicción (escrita antes de los experimentos):** con 100 % de ruido el test queda en el azar (≈10 %) aunque el train suba. El tiempo hasta memorizar crece con el ruido. Con ruido parcial, el test sube primero y baja después, porque las redes aprenden antes los patrones reales que el ruido (Arpit et al., 2017). Sobre el ritmo: ningún modelo con sesgo memoriza antes que el MLP, porque su sesgo encaja con las etiquetas reales y no con las aleatorias, y el MLP no tiene ninguna estructura que le estorbe.
"""), code("""
if TRAIN:
    E.h4(data, store, models=ONLY_MODELS, seeds=ONLY_SEEDS)
h4s, h4c = store.rows("h4"), store.rows("h4curves")
display(T.h4_table(h4s, "epoch_95"))   # épocas hasta el 95 % de train (media ± σ de las semillas que llegan)
fig = Fg.h4(h4s, h4c); Fg.save(fig, "h4"); plt.show()
# oscilación de la precisión de train entre épocas consecutivas (media de |Δ| tras la época 20), con 50 % de ruido
c50 = h4c[(h4c.label_noise == 0.5)].sort_values("epoch")
osc = c50.groupby(["model", "weight_decay", "seed"]).train_acc.apply(lambda s: s.diff().abs().iloc[20:].mean()).reset_index()
display(T.pm_table(osc, "weight_decay", "model", "train_acc",
                   index_name="weight decay · oscilación media de la precisión de train entre épocas (puntos), 50 % al azar"))
"""), md("""
**Resultado: confirmada.** Las cuatro arquitecturas memorizan etiquetas totalmente aleatorias y lo hacen a ritmos muy distintos. El MLP necesita un learning rate menor que el del protocolo para conseguirlo (control al final de esta sección).

Épocas hasta superar el 95 % de precisión en train, sin weight decay (media de 5 semillas; tabla completa con σ arriba). El «pico de test» de la tabla es descriptivo: se elige mirando el test, así que no es una estimación de generalización.

| Etiquetas aleatorias | MLP | CNN | LSTM | Transformer |
|---|---|---|---|---|
| 0 % | 17 | 32 | 29 | 28 |
| 25 % | 64 | 93 | 64 | 48 |
| 50 % | 85 | 105 | 68 | 50 |
| 100 % | no llega (0/5); 110 con lr 10⁻³ | 117 | 73 | 59 |

- **Se cumplen las tres predicciones sobre el test.** Con el 100 % de etiquetas aleatorias el test termina en el azar (entre el 10,1 y el 10,4 %) aunque el train llegue al 100 % (en el MLP, con el learning rate del protocolo, al 81 %). Sin weight decay, el tiempo hasta memorizar crece con el ruido en todos los modelos. Y con ruido parcial el test sube primero y baja después: con el 50 % de ruido, el máximo y el valor final son 78,7 y 51,8 % en el MLP, 74,6 y 53,4 % en la CNN, 78,5 y 53,7 % en la LSTM y 72,6 y 47,6 % en el Transformer. Las redes aprenden antes los patrones reales que el ruido (Arpit et al., 2017).
- **El sesgo inductivo no protege contra la memorización.** La CNN, la LSTM y el Transformer memorizan el 100 % de etiquetas aleatorias en 117, 73 y 59 épocas, como en Zhang et al. (2017). Con etiquetas ruidosas, el Transformer es el más rápido en llegar al 95 %.
- **El MLP no llega con el learning rate del protocolo, pero sí con uno menor.** Con 3·10⁻³ se estanca en torno al 81 % de train (entre el 70 y el 88 % según la semilla). Tiene parámetros de sobra (101 706 para 5 000 imágenes), así que sospechábamos de la optimización, y el control lo confirma: con 10⁻³ memoriza en 110 ± 6 épocas y con 3·10⁻⁴ en 148 ± 5, las cinco semillas. 3·10⁻³ se eligió con etiquetas limpias, donde el MLP es el más rápido de los cuatro (17 épocas); con etiquetas aleatorias es demasiado alto para ajustar los últimos ejemplos.
- **El weight decay retrasa la memorización, pero no la impide ni mejora el test.** Con 0,1 y 50 % de ruido, el MLP pasa de 85 a 121 épocas, la LSTM de 68 a 86 y el Transformer de 50 a 56, y el máximo en test cambia menos de un punto (con weight decay, la LSTM tarda algo más con el 50 % de ruido que con el 100 %: 86 frente a 81 épocas). Coincide con Zhang et al. (2017): la regularización explícita no es lo que explica la generalización.
- **La CNN es la excepción con weight decay, y por un motivo de medida y optimización.** Con 0,1 no llega al 95 % con el 50 % de ruido (1 de 5 semillas) ni con el 100 % (0 de 5). Su precisión de train, medida en modo evaluación, oscila 13-15 puntos de una época a la siguiente en todas las condiciones con ruido, también sin weight decay, frente a 1-2,5 puntos en el resto (tabla de arriba). Es la única con BatchNorm, y el control lo explica: medida durante el entrenamiento, con las estadísticas de cada lote, la oscilación baja a unos 2 puntos, como en los demás. Es decir, la red aprende de forma estable, pero las estadísticas acumuladas de BatchNorm que usa la evaluación van por detrás de unos pesos que cambian mucho con un learning rate alto. Con weight decay se agrava, porque con BatchNorm encoger los pesos sube el learning rate efectivo (van Laarhoven, 2017).

Respecto a la predicción previa sobre el ritmo (ningún modelo con sesgo memorizaría antes que el MLP): falla. El Transformer es más rápido que el MLP con cualquier nivel de ruido; la LSTM, con el 50 y el 100 % (con el 25 % empatan: 64 frente a 64 épocas); y la CNN es más lenta con ruido parcial y más rápida con etiquetas totalmente aleatorias. Con el mismo learning rate para todos (10⁻³, control siguiente) el orden con el 100 % de ruido es Transformer (59 épocas), LSTM (81), MLP (110) y CNN (199): la predicción falla para el Transformer y la LSTM y se cumple para la CNN.
"""), md("""
**Controles: ¿es la arquitectura o el learning rate?** Tras ver los resultados repetimos algunas configuraciones con otro learning rate constante, sin cambiar nada más. Se ejecutaron en CPU con `python -m sesgos.run extra`; la fila «protocolo» es la ejecución principal. El MLP con 3·10⁻³ se repitió también en CPU para comprobar que el dispositivo no cambia la conclusión.
"""), code("""
if TRAIN:
    E.h4_control(data, store)
ctl = store.rows("h4ctl")
if len(ctl):
    display(T.h4_control_table(ctl, h4s))
    cc = store.rows("h4ctlcurves")
    q = cc[(cc.model == "cnn") & (cc.label_noise == 0.5) & np.isclose(cc.lr, 3e-3)].sort_values("epoch")
    if len(q) and "train_acc_batch" in q:
        osc = q.groupby("seed").apply(lambda g: pd.Series({
            "medida en evaluación": g.train_acc.diff().abs().iloc[20:].mean() * 100,
            "medida durante el entrenamiento": g.train_acc_batch.diff().abs().iloc[20:].mean() * 100}))
        display(osc.agg(["mean", "std"]).T.map(lambda v: f"{v:.1f}".replace(".", ",")).rename(columns={"mean": "media", "std": "σ"})
                .rename_axis("CNN, 50 % al azar · oscilación de la precisión de train entre épocas (puntos)"))
else:
    print("Sin controles en results/raw (h4ctl).")
"""), md("""
**Resultado de los controles.**

- **MLP:** con 10⁻³ memoriza el 100 % de etiquetas aleatorias en 110 ± 6 épocas y con 3·10⁻⁴ en 148 ± 5, las cinco semillas. Con 3·10⁻³, repetido en CPU, 1 de 5 semillas llega (en la época 250) y el resto se queda entre el 71 y el 92 %: el dispositivo no cambia la conclusión. El estancamiento era del learning rate, no de la arquitectura.
- **LSTM y CNN con 10⁻³ (3 semillas):** la LSTM memoriza en 81 épocas (73 con 3·10⁻³) y la CNN en 199 ± 28 (117 con 3·10⁻³). Con el mismo learning rate para los cuatro, el orden es Transformer (59), LSTM (81), MLP (110) y CNN (199). El ritmo depende de la arquitectura, y la diferencia entre la más rápida y la más lenta se agranda: de 2 a más de 3 veces.
- **CNN, oscilación:** con el 50 % de ruido, la precisión de train medida en evaluación salta 13,5 puntos de media entre épocas (3 semillas) y medida durante el entrenamiento, 2,0, como los demás modelos. La oscilación es de la medida con las estadísticas acumuladas de BatchNorm, no del aprendizaje. (La medida durante el entrenamiento promedia toda la época, así que es algo más suave por construcción; aun así, la diferencia es de casi siete veces.)
""")]

cells += [md("""
## 8. Explicabilidad

**E1. Pesos de la primera capa (H1).** Filtros 3×3 de la CNN entrenada con imágenes originales y con píxeles permutados. Y filas de la primera capa del MLP: original, y entrenado con píxeles permutados después de deshacer la permutación. Si el MLP es invariante, sus plantillas deben parecerse; la CNN permutada no debería tener filtros con estructura espacial. Para no depender de la vista, lo medimos con la correlación entre pesos vecinos (horizontal y vertical) de la primera capa, en las cinco semillas: alta si los pesos varían suavemente en el espacio, cerca de 0 si no hay estructura.
"""), code("""
def have(*pairs):
    ok = all(os.path.exists(store.ckpt(m, c, 0)) for m, c in pairs)
    if not ok:
        print("Faltan checkpoints para esta figura:", pairs)
    return ok

def load(model, cond, seed=0):
    m = build(model).to(DEVICE)
    m.load_state_dict(torch.load(store.ckpt(model, cond, seed), map_location=DEVICE))
    return m.eval()

perm = pixel_permutation(C.PERM_SEED)
inv = torch.argsort(perm)
if have(("cnn", "original"), ("cnn", "pixeles")):
    f = [X.first_layer_images(load("cnn", c), "cnn").cpu().numpy()[:10] for c in ["original", "pixeles"]]
    fig = Fg.image_grid([*f[0], *f[1]], 2, 10, title="CNN: filtros 3×3 de la primera capa (semilla 0)",
                        row_labels=["original", "píxeles\\npermutados"], cbar_label="peso / máx. |peso|")
    Fg.save(fig, "e1_cnn"); plt.show()
if have(("mlp", "original"), ("mlp", "pixeles")):
    w0 = X.first_layer_images(load("mlp", "original"), "mlp").cpu().numpy()
    wp = X.first_layer_images(load("mlp", "pixeles"), "mlp", inverse_perm=inv.to(DEVICE)).cpu().numpy()
    wr = X.first_layer_images(load("mlp", "pixeles"), "mlp").cpu().numpy()
    fig = Fg.image_grid(list(w0[:8]) + list(wp[:8]) + list(wr[:8]), 3, 8, figsize=(9.5, 4.0),
                        title="MLP: plantillas de la primera capa (semilla 0)",
                        row_labels=["original", "permutado,\\ndeshecha la\\npermutación", "permutado,\\ntal cual"],
                        cbar_label="peso / máx. |peso|")
    Fg.save(fig, "e1_mlp"); plt.show()
# Medida: correlación entre pesos vecinos de la primera capa, con todas las semillas cuyos checkpoints haya
E.e1_smoothness(data, store)
e1 = store.rows("e1")
e1v = e1.assign(corr=lambda d: d.neighbor_corr_despermutado.fillna(d.neighbor_corr))
display(T.pm_table(e1v, "condition", "model", "corr", scale=1, dec=3,
                   index_name="correlación entre pesos vecinos, media ± σ (5 semillas; MLP permutado: deshecha la permutación)")
        .reindex(["original", "bloques", "pixeles"]).fillna("—"))
"""), md("""
**Resultado.** Las figuras son de la semilla 0 y se leen de forma cualitativa; la medida usa las cinco. Las plantillas del MLP son igual de suaves con y sin permutación, una vez deshecha: la correlación entre pesos vecinos es 0,559 ± 0,006 en el original y 0,557 ± 0,006 en el permutado y des-permutado (0,001 sin deshacer la permutación). Es la versión en los pesos del resultado de H1: el MLP aprende lo mismo, en otro orden. En la CNN la diferencia va en la dirección esperada pero es pequeña: 0,06 ± 0,06 en el original frente a −0,09 ± 0,03 con píxeles permutados (diferencia −0,15, IC 95 % [−0,22, −0,08]) y 0,02 con bloques. Con filtros de 3×3 seguidos de BatchNorm, la primera capa de la CNN no muestra detectores de bordes nítidos a simple vista, así que la figura de sus filtros, por sí sola, no es concluyente.
"""), md("""
**E2. ¿Se mueve la explicación con la prenda? (H3).** Integrated Gradients (Sundararajan et al., 2017) con la imagen en negro como referencia. Primero, una prenda sin desplazar y desplazada 4 píxeles a la derecha en los cuatro modelos. Después, la medida: para 300 imágenes de test, la correlación entre el mapa de la imagen desplazada *k* píxeles y el mapa original desplazado *k* píxeles. Vale 1 si la explicación se mueve exactamente con la prenda y 0 si no tiene nada que ver.
"""), code("""
i = 1                                          # la misma prenda que en el póster
x = data.Xte[i:i + 1]; y = data.Yte[i:i + 1]
names = [m for m in C.MODELS if have((m, "original"))]
maps = {k: [] for k in [0, 4]}
for m_name in names:
    m = load(m_name, "original")
    for k in [0, 4]:
        xi = data.norm(shift(x, dx=k))
        maps[k].append(X.integrated_gradients(m, xi, y, data.norm(torch.zeros_like(x)))[0].cpu().numpy())
fig = Fg.image_grid(maps[0] + maps[4], 2, len(names), ax_titles=[C.MODEL_LABELS[n] for n in names] + [""] * len(names),
                    figsize=(7.5, 4.0), row_labels=["original", "desplazada\\n4 px"], cbar_label="atribución / máx.",
                    title=f"Integrated Gradients, clase {CLASSES[int(y)]} (semilla 0)")
Fg.save(fig, "e2_saliency"); plt.show()
E.e2_equivariance(data, store)
e2 = store.rows("e2")
fig = Fg.e2(e2); Fg.save(fig, "e2_equivarianza"); plt.show()
for d in ["horizontal", "vertical"]:
    display(T.pm_table(e2[e2.direction == d], "shift", "model", "corr", scale=1, dec=2,
                       index_name=f"desplazamiento {d} (px) · correlación, media ± σ"))
"""), md("""
**Resultado.** La explicación de la CNN se desplaza con la prenda y la de los demás modelos no, como predice H3:

- **CNN:** 0,91 con 4 píxeles en horizontal y 0,87 en vertical. La curva tiene forma de sierra: en torno a 0,4 con desplazamientos impares, 0,7 con múltiplos de 2 y entre 0,80 y 0,91 con múltiplos de 4. Es la huella de sus dos max-pool de 2×2: la red solo es equivariante a desplazamientos múltiplos de su paso acumulado, que es 4 (Azulay y Weiss, 2019).
- **LSTM:** se mueve en vertical (0,85 con 1 píxel y 0,48 con 4) y no en horizontal (−0,01 con 4), porque un desplazamiento vertical es un retraso de la secuencia de filas.
- **Transformer:** casi 0 en las dos direcciones, con un pequeño repunte con 4 y 8 píxeles (0,09 y 0,08 en horizontal; 0,23 y 0,11 en vertical), cuando el desplazamiento mueve parches enteros. Sus posiciones son absolutas.
- **MLP:** 0,38 y 0,49 con 1 píxel, solo porque los mapas son algo suaves, y 0,12 o menos a partir de 4 píxeles.

La tabla de H3 y esta medida apuntan en la misma dirección: el modelo que conserva la precisión al desplazar la prenda es también el que desplaza su explicación con ella. Es una coincidencia entre medidas, no una prueba de que la explicación cause la tolerancia; la LSTM, por ejemplo, retiene el 86 % en vertical con una explicación que se mueve a medias (0,48).
"""), md("""
**E3. Atención del Transformer.** Atención del token CLS sobre los 49 parches en el último bloque, en imágenes originales y con los bloques permutados (deshaciendo la permutación para compararlas).
"""), code("""
if have(("vit", "original"), ("vit", "bloques")):
  v = load("vit", "original")
  bperm, order = block_permutation(C.PERM_SEED)
  xs = data.Xte[:6]
  a0 = X.vit_cls_attention(v, data.norm(xs))[-1].cpu()
  vb = load("vit", "bloques")
  ab = X.vit_cls_attention(vb, data.norm(apply_perm(xs, bperm)))[-1].cpu().reshape(-1, 49)
  ab = ab[:, torch.argsort(order)].reshape(-1, 7, 7)
  fig = Fg.image_grid([*xs[:, 0].cpu().numpy(), *a0.numpy(), *ab.numpy()], 3, 6, cmap="magma", symmetric=False,
                      figsize=(8.5, 4.4), title="Transformer: atención del token CLS, último bloque (semilla 0)",
                      row_labels=["imagen", "atención\\n(original)", "atención\\n(bloques,\\ndeshecha la\\npermutación)"],
                      cbar_label="atención / máx.")
  Fg.save(fig, "e3_attention"); plt.show()
"""), md("""
**Observación (cualitativa, semilla 0).** En el último bloque, el token CLS atiende a pocos parches, sobre todo en los bordes de la prenda. Con los bloques permutados y la permutación deshecha, la atención también cae sobre parches de la prenda, pero no en los mismos: son dos modelos entrenados por separado, con precisión parecida (88,1 y 88,6 % en esta semilla). Es coherente con el control exacto de H1, según el cual permutar bloques equivale a permutar posiciones, pero no lo demuestra: la prueba es ese control, no esta figura.
""")]

cells += [md("""
## 9. Conclusiones

**Un sesgo inductivo ayuda cuando los datos tienen la estructura que el sesgo supone, y puede perjudicar cuando no la tienen.** La CNN es la mejor con imágenes reales y queda por debajo del MLP con los píxeles desordenados; tolera desplazamientos porque la convolución los respeta y no tolera giros porque nada en ella los respeta.

| Hipótesis | Veredicto | Por qué |
|---|---|---|
| H1 · Estructura de la entrada | Confirmada | Con los píxeles permutados la CNN pierde 3,9 puntos y la LSTM 2,8, y el MLP no cambia. El Transformer solo cae con píxeles: permutar bloques equivale a permutar sus posiciones, que vuelve a aprender. |
| H2 · Cantidad de datos | Confirmada, con matiz | Con 100 imágenes el mejor y el peor modelo se separan 18,6 puntos (3,6 con 55 000), pero porque la LSTM y el Transformer se hunden. La CNN no supera al MLP con 100 imágenes (tampoco cambiando el subconjunto), empieza a superarlo entre 300 y 1 000 y su ventaja crece con los datos. |
| H3 · Transformaciones en test | Confirmada | Desplazada 4 píxeles, la CNN retiene el 95 y el 92 % de su precisión, frente al 30 y el 60 % del MLP. La LSTM solo tolera el desplazamiento vertical. Ninguna tolera la rotación. |
| H4 · Memorización | Confirmada | Las cuatro memorizan etiquetas totalmente aleatorias con el test en el azar, a ritmos muy distintos: 59 épocas el Transformer, 73 la LSTM, 117 la CNN y 110 el MLP (este, con learning rate 10⁻³; con el del protocolo se estanca). Con 10⁻³ para todos, el orden es Transformer, LSTM, MLP y CNN. El weight decay retrasa la memorización sin mejorar el test. |

**Límites del estudio.**

- Un solo conjunto de datos, con prendas centradas y escaladas, y modelos de unos 100 000 parámetros: no sabemos si las conclusiones se mantienen con imágenes menos regulares o con modelos más grandes.
- El learning rate se eligió una vez por modelo, con todos los datos, en la rejilla {3·10⁻⁴, 10⁻³, 3·10⁻³}: 3·10⁻³ para el MLP, la CNN y la LSTM, que es el extremo de la rejilla, y 10⁻³ para el Transformer. Un control con 10⁻² no mejora al MLP (89,4 % en validación frente a 90,4 %) ni a la LSTM (91,2 frente a 91,3 %), pero sí a la CNN (93,6 frente a 92,9 %, una semilla): su ventaja sobre el MLP está, si acaso, subestimada. Se mantiene en H2 y en H4; el estancamiento del MLP con etiquetas aleatorias venía de ese learning rate (controles de H4).
- Una sola técnica de regularización (weight decay) con una sola intensidad (0,1).
- En H2 el subconjunto de cada tamaño es el mismo en todas las semillas; el control con otro subconjunto por semilla mantiene la conclusión con 100 imágenes y retrasa el cruce de la CNN con el MLP.
- Cinco semillas por condición y muchas comparaciones: las diferencias pequeñas con p cercano a 0,05 no deben leerse como efectos. Las caídas de H1, la tolerancia de H3 y el ritmo de memorización de H4 tienen p < 0,001; la desventaja de la CNN frente al MLP con 100 imágenes es más débil (p = 0,016 con el subconjunto del protocolo y 0,038 emparejada con otros subconjuntos), y por eso se apoya en los dos.

Las predicciones se escribieron antes de los experimentos y no se han cambiado. Los veredictos de la figura siguiente usan los criterios escritos en `verdicts.py`.

La figura siguiente responde una pregunta por hipótesis para cada modelo (sí, en parte o no): círculo hueco para lo predicho antes de los experimentos y relleno para lo observado. El MLP es la referencia en H2 y H4, así que esas celdas quedan vacías.
"""), code("""
from sesgos.verdicts import PRED, OBS      # PRED es la predicción previa; OBS se rellena en sesgos/verdicts.py
fig = Fg.verdict_matrix(PRED, OBS); Fg.save(fig, "veredictos"); plt.show()
"""), md("""
## 10. Figuras para el póster

Las mismas figuras con letra a escala de póster, exportadas al ancho exacto de su hueco en el A0, en PDF vectorial (`figures/poster/`). También se pueden generar fuera del notebook con `python tools/make_figures.py --results <carpeta>`.
"""), code("""
print(Fg.all_figures(RESULTS, "figures"))
"""), md("""
## Referencias

- Xiao, Rasul y Vollgraf (2017). Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms. arXiv:1708.07747.
- Zhang, Bengio, Hardt, Recht y Vinyals (2017). Understanding deep learning requires rethinking generalization. ICLR.
- Arpit et al. (2017). A Closer Look at Memorization in Deep Networks. ICML.
- Azulay y Weiss (2019). Why do deep convolutional networks generalize so poorly to small image transformations? JMLR 20.
- Kayhan y van Gemert (2020). On Translation Invariance in CNNs: Convolutional Layers Can Exploit Absolute Spatial Location. CVPR.
- Dosovitskiy et al. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. ICLR.
- Sundararajan, Taly y Yan (2017). Axiomatic Attribution for Deep Networks. ICML.
- van Laarhoven (2017). L2 Regularization versus Batch and Weight Normalization. arXiv:1706.05350.
""")]

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
               "language_info": {"name": "python"}, "accelerator": "GPU", "colab": {"provenance": []}}
out = os.path.join(ROOT, "sesgos_inductivos.ipynb")
nbf.write(nb, out)
print("escrito", out, len(cells), "celdas")
