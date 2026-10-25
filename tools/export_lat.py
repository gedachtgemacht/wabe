"""Build the lattices natively and store them as web/py/lat.pkl. Needed because build_lattice decides marker-edge cases
(distance == radius) by floating-point noise from sin/cos: in WebAssembly (musl) this falls differently than natively
(glibc), and one cell of difference shifts the whole bit sequence. Add new radii/shapes here and run again.
Usage (from the project folder): python tools/export_lat.py"""
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
