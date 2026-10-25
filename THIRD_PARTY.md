# Drittcode

* `web/py/reedsolo.py`: reedsolo 1.7.0 (Tomer Filiba, Stephen Larroque), Public Domain / Unlicense / MIT-0, unveraendert uebernommen.
  Der Lokal-Modus laedt es in Pyodide; der Server nutzt das PyPI-Paket `reedsolo`.
* Nicht im Repo, wird zur Laufzeit bzw. mit `tools/pyodide_holen.sh` geladen: Pyodide 0.27.7 (MPL-2.0), NumPy (BSD-3-Clause),
  OpenCV (Apache-2.0), Brotli (MIT).
* Python-Abhaengigkeiten des Servers: opencv-python-headless, numpy, reedsolo, brotli (siehe README).
