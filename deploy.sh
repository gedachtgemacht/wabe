#!/bin/bash
# wabe. auf den Hetzner bringen: Server, Seite, Lokal-Modus (py/, pyodide/) und Service Worker.
# Aufruf: ./deploy.sh   (aus ~/Schreibtisch/wabe; Zugang per ssh hetzner)
set -e
cd "$(dirname "$0")"
mkdir -p web/py
cp wabe.py dicht.py strom.py web/py/                       # Lokal-Modus nutzt dieselben Quellen wie der Server
V=$(cat server.py web/index.html web/lokal.js web/py/*.py web/py/lat.pkl wabe.py | md5sum | cut -c1-10)
sed "s/^const VERSION = .*/const VERSION = '$V';/" web/sw.js > /tmp/wabe-sw.js
scp -q wabe.py dicht.py strom.py server.py hetzner:/opt/wabe/
scp -q web/index.html web/lokal.js web/manifest.webmanifest web/icon-192.png web/icon-512.png hetzner:/opt/wabe/web/
scp -q /tmp/wabe-sw.js hetzner:/opt/wabe/web/sw.js
ssh hetzner "mkdir -p /opt/wabe/web/py /opt/wabe/web/pyodide"
scp -q web/py/* hetzner:/opt/wabe/web/py/
rsync -q -rt web/pyodide/ hetzner:/opt/wabe/web/pyodide/
ssh hetzner "chmod -R a+rX /opt/wabe; systemctl restart wabe; sleep 2; systemctl is-active wabe; curl -s http://127.0.0.1:8915/api/health"
echo "Version $V"
