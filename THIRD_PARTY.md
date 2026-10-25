# Third-party code

* `web/py/reedsolo.py`: reedsolo 1.7.0 (Tomer Filiba, Stephen Larroque), Public Domain / Unlicense / MIT-0, included unchanged.
  The in-browser mode loads it into Pyodide; the server uses the PyPI package `reedsolo`.
* Not in the repo, downloaded at runtime or with `tools/pyodide_holen.sh`: Pyodide 0.27.7 (MPL-2.0), NumPy (BSD-3-Clause),
  OpenCV (Apache-2.0), Brotli (MIT).
* Python dependencies of the server: opencv-python-headless, numpy, reedsolo, brotli (see the README).
