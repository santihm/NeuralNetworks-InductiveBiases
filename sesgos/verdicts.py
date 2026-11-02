"""Matriz de veredictos del póster. Preguntas por hipótesis:
H1 ¿cae al permutar?  H2 ¿supera al MLP con pocos datos?  H3 ¿tolera la traslación?  H4 ¿memoriza antes que el MLP?
Valores: "si", "parcial", "no". El MLP es la referencia en H2 y H4, así que no tiene celda ahí."""

# Predicción escrita antes de los experimentos (plan del 18 de octubre). No se modifica.
PRED = {("mlp", "H1"): "no", ("cnn", "H1"): "si", ("lstm", "H1"): "si", ("vit", "H1"): "parcial",
        ("cnn", "H2"): "si", ("lstm", "H2"): "parcial", ("vit", "H2"): "no",
        ("mlp", "H3"): "no", ("cnn", "H3"): "si", ("lstm", "H3"): "parcial", ("vit", "H3"): "no",
        ("cnn", "H4"): "no", ("lstm", "H4"): "no", ("vit", "H4"): "no"}

# Observado con las cinco semillas (protocolo v2). Criterios, fijados al rellenar la tabla:
# H1: caída significativa (IC 95 % de Welch por debajo de 0) con píxeles y con bloques = "si"; con una sola = "parcial".
# H2: supera al MLP (IC 95 % por encima de 0) con 100, 300 y 1 000 imágenes = "si"; en alguno de esos tamaños = "parcial".
# H3: retiene al menos el 85 % de su precisión con 4 px en las dos direcciones = "si"; en una sola, o claramente por
#     encima del MLP en las dos (más de 10 puntos, IC por encima de 0) sin llegar al 85 % = "parcial".
# H4: llega al 95 % de train antes que el MLP (sin weight decay, IC por debajo de 0, o el MLP no llega) con 25, 50 y
#     100 % de etiquetas al azar = "si"; solo con alguno = "parcial".
OBS = {("mlp", "H1"): "no", ("cnn", "H1"): "si", ("lstm", "H1"): "si", ("vit", "H1"): "parcial",
       ("cnn", "H2"): "parcial", ("lstm", "H2"): "no", ("vit", "H2"): "no",
       ("mlp", "H3"): "no", ("cnn", "H3"): "si", ("lstm", "H3"): "parcial", ("vit", "H3"): "parcial",
       ("cnn", "H4"): "parcial", ("lstm", "H4"): "parcial", ("vit", "H4"): "si"}
