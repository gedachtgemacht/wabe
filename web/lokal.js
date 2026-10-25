// wabe. local mode: decoder in Pyodide (WebAssembly), runs in this worker. Messages:
//   {t:'decode', id, w, h, buf}  -> {t:'res', id, r}      (r = measurements + raw payload bytes as base64, stateless)
//   {t:'add', id, payloads}      -> {t:'res', id, r, data?} (stream solver, worker 0 only; data = finished file)
//   {t:'neu'}                    -> reset the stream solver
// Status goes to the main thread via {t:'status', text}.
importScripts('pyodide/pyodide.js');                       // Pyodide is self-hosted on the server (and in the service-worker cache)
let py = null, bereit = null;

async function start() {
  const t0 = performance.now();
  postMessage({ t: 'status', text: 'Loading decoder …' });
  py = await loadPyodide({ indexURL: new URL('pyodide/', self.location.href).href });
  postMessage({ t: 'status', text: 'Loading packages …' });
  await py.loadPackage(['numpy', 'opencv-python', 'brotli', 'lzma']);
  for (const f of ['wabe.py', 'dicht.py', 'strom.py', 'reedsolo.py', 'lokal.py', 'lat.pkl']) {
    const r = await fetch('py/' + f);
    if (!r.ok) throw new Error('py/' + f + ' ' + r.status);
    py.FS.writeFile(f, new Uint8Array(await r.arrayBuffer()));
  }
  postMessage({ t: 'status', text: 'Starting decoder …' });
  await py.runPythonAsync('import lokal');
  postMessage({ t: 'status', text: '' });
  postMessage({ t: 'bereit', ms: Math.round(performance.now() - t0) });
}
bereit = start().catch(e => postMessage({ t: 'fehler', text: String(e) }));

function nachJs(res) { const r = res.toJs({ dict_converter: Object.fromEntries }); res.destroy(); return r; }

onmessage = async ev => {
  const m = ev.data;
  await bereit;
  if (!py) return;
  if (m.t === 'neu') { py.runPython('lokal.neu()'); return; }
  let r, data = null;
  try {
    if (m.t === 'decode') {
      py.globals.set('_buf', new Uint8Array(m.buf));
      py.globals.set('_w', m.w); py.globals.set('_h', m.h);
      r = nachJs(py.runPython('lokal.decode(_buf.to_py(), _w, _h)'));
    } else if (m.t === 'add') {
      py.globals.set('_p', m.payloads);
      r = nachJs(py.runPython('lokal.add(_p.to_py())'));
      if (r.fertig) data = py.runPython('from pyodide.ffi import to_js; to_js(lokal.datei)');
    }
  } catch (e) { r = { ok: false, fehler: String(e).slice(0, 160), payloads: [] }; }
  postMessage({ t: 'res', id: m.id, r, data }, data ? [data.buffer] : []);
};
