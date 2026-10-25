"""Gitter nativ bauen und als web/py/lat.pkl ablegen. Noetig, weil build_lattice Markerrand-Faelle (Abstand == Radius)
nach dem Gleitkomma-Rauschen von sin/cos entscheidet: in WebAssembly (musl) faellt das anders als nativ (glibc),
eine Zelle Unterschied verschiebt die ganze Bitfolge. Fuer neue Radien/Formen hier ergaenzen und neu ausfuehren.
Aufruf (aus dem Projektordner): python tools/export_lat.py"""
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import wabe  # noqa: E402

out = {}
for R, shape in ((100, "rect"), (128, "rect"), (100, "circle"), (128, "circle"), (128, "hexagon")):
    lat = wabe.build_lattice(R, shape)
    out[f"{shape}{R}"] = dict(fixed=lat.fixed, data_idx=lat.data_idx.astype(np.int32),
                              format_idx=lat.format_idx.astype(np.int32),
                              markers=[(int(a), int(b), (float(c[0]), float(c[1]))) for a, b, c in lat.markers])
pickle.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "web", "py", "lat.pkl"), "wb"), protocol=4)
print("web/py/lat.pkl:", ", ".join(out))
