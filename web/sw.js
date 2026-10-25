// wabe. service worker: turns the page into an app that also runs without network (local mode).
// deploy.sh sets VERSION on every deploy; a new value reloads the page, lokal.js and py/.
const VERSION = 'dev';
const SHELL = 'wabe-shell-' + VERSION;
const PYO = 'wabe-pyodide-v1';                  // only changes with the Pyodide version
const SHELL_FILES = ['./', 'index.html', 'lokal.js', 'manifest.webmanifest', 'icon-192.png', 'icon-512.png',
  'py/wabe.py', 'py/dicht.py', 'py/strom.py', 'py/reedsolo.py', 'py/lokal.py', 'py/lat.pkl'];
const PYO_FILES = ['pyodide/pyodide.js', 'pyodide/pyodide.asm.js', 'pyodide/pyodide.asm.wasm', 'pyodide/python_stdlib.zip',
  'pyodide/pyodide-lock.json',
  'pyodide/numpy-2.0.2-cp312-cp312-pyodide_2024_0_wasm32.whl',
  'pyodide/opencv_python-4.10.0.84-cp312-cp312-pyodide_2024_0_wasm32.whl',
  'pyodide/brotli-1.1.0-cp312-cp312-pyodide_2024_0_wasm32.whl',
  'pyodide/lzma-1.0.0-cp312-cp312-pyodide_2024_0_wasm32.whl'];

async function meldung(m) {
  for (const c of await self.clients.matchAll({ includeUncontrolled: true })) c.postMessage(m);
}

async function vorladen() {
  const shell = await caches.open(SHELL), pyo = await caches.open(PYO);
  const alle = [...SHELL_FILES.map(f => [shell, f, true]), ...PYO_FILES.map(f => [pyo, f, false])];
  let done = 0;
  for (const [cache, f, frisch] of alle) {
    try {
      if (!frisch && await cache.match(f)) { /* Pyodide already cached */ }
      else {
        const r = await fetch(f, { cache: frisch ? 'reload' : 'default' });
        if (r.ok) await cache.put(f, r);
      }
    } catch (e) { /* offline or error: fetch on first use instead */ }
    meldung({ t: 'sw', done: ++done, total: alle.length, fertig: false });
  }
  const fehlt = [];
  for (const [cache, f] of alle) if (!(await cache.match(f))) fehlt.push(f);
  meldung({ t: 'sw', done, total: alle.length, fertig: fehlt.length === 0, fehlt });
}

self.addEventListener('install', ev => { self.skipWaiting(); ev.waitUntil(vorladen()); });

self.addEventListener('activate', ev => {
  ev.waitUntil((async () => {
    for (const k of await caches.keys()) if (k.startsWith('wabe-shell-') && k !== SHELL) await caches.delete(k);
    await self.clients.claim();
  })());
});

self.addEventListener('fetch', ev => {
  const req = ev.request, url = new URL(req.url);
  if (req.method !== 'GET' || url.origin !== location.origin || url.pathname.includes('/api/')) return;
  const pyodide = url.pathname.includes('/pyodide/');
  ev.respondWith((async () => {
    const cache = await caches.open(pyodide ? PYO : SHELL);
    const key = req.mode === 'navigate' ? 'index.html' : req;
    const hit = await cache.match(key, { ignoreSearch: true });
    // Pyodide: from the cache. Everything else: cache first, refresh in the background (the next start has the new version).
    const holen = fetch(req).then(r => { if (r.ok && !pyodide) cache.put(key, r.clone()); else if (r.ok) cache.put(req, r.clone()); return r; });
    if (hit) { if (!pyodide) holen.catch(() => {}); return hit; }
    return holen;
  })());
});
