#!/bin/bash
# Download Pyodide 0.27.7 (runtime + the four packages of the in-browser mode) into web/pyodide/; the files are not in the repo.
set -e
cd "$(dirname "$0")/.." && mkdir -p web/pyodide && cd web/pyodide
B=https://cdn.jsdelivr.net/pyodide/v0.27.7/full
for f in pyodide.js pyodide.asm.js pyodide.asm.wasm python_stdlib.zip pyodide-lock.json \
  numpy-2.0.2-cp312-cp312-pyodide_2024_0_wasm32.whl opencv_python-4.10.0.84-cp312-cp312-pyodide_2024_0_wasm32.whl \
  brotli-1.1.0-cp312-cp312-pyodide_2024_0_wasm32.whl lzma-1.0.0-cp312-cp312-pyodide_2024_0_wasm32.whl; do
  [ -s "$f" ] || curl -sfL -o "$f" "$B/$f"
done
ls -la
