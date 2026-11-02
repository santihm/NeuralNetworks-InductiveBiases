"""Genera las figuras del notebook y del póster desde una carpeta de resultados (por ejemplo, la del Drive).
Uso: python tools/make_figures.py --results "<ruta>/results" [--out figures]"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sesgos.figures import all_figures

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results")
ap.add_argument("--out", default="figures")
a = ap.parse_args()
print("\n".join(all_figures(a.results, a.out)))
