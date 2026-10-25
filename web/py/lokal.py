"""wabe. local mode: decoder in the browser (Pyodide). The lattices (fixed/data_idx/...) come from lat.pkl, built natively:
build_lattice decides marker-edge cases (distance == radius) by floating-point noise from sin/cos, which in
WebAssembly (musl) falls differently than natively (glibc); one cell of difference shifts the whole bit sequence."""
import base64
import pickle
import struct
import time

import numpy as np

import strom
import wabe

LAT = pickle.load(open("lat.pkl", "rb"))
_orig = wabe.build_lattice


def _bl(R, shape):
    if (R, shape) in wabe._LAT_CACHE:
        return wabe._LAT_CACHE[(R, shape)]
    lat = _orig(R, shape)
    d = LAT.get(f"{shape}{R}")
    if d is not None and len(d["fixed"]) == len(lat.fixed):
        lat.fixed = d["fixed"]
        lat.data_idx = d["data_idx"].astype(np.int64)
        lat.format_idx = d["format_idx"].astype(np.int64)
        lat.markers = d["markers"]
    return lat


wabe.build_lattice = _bl

dec = strom.StreamDecoder()
datei = b""


def neu():
    global dec, datei
    dec = strom.StreamDecoder()
    datei = b""


def decode(buf, w, h):
    """RGBA bytes of a camera image -> raw payload bytes of all readable codes (base64) plus measurements; stateless."""
    t = time.time()
    img = np.ascontiguousarray(np.frombuffer(buf, np.uint8).reshape(h, w, 4)[:, :, 2::-1])
    out = {"ok": False, "payloads": []}
    try:
        found = wabe.decode_all(img, parse=False)
        out.update(ok=True, payloads=[base64.b64encode(raw).decode() for raw, _ in found],
                   ppm=round(max(i.ppm / i.scale for _, i in found), 2), rs=max(i.rs_errors for _, i in found))
    except Exception as e:
        out["fehler"] = (str(e) or type(e).__name__)[:160]
    out["sek"] = round(time.time() - t, 2)
    return out


def add(payloads):
    """Feed payload bytes to the stream solver (runs in worker 0 only) -> progress."""
    global datei
    gelesen, fehler = 0, None
    for b in payloads:
        try:
            dec.add_payload(base64.b64decode(b))
            gelesen += 1
        except (ValueError, struct.error) as e:
            fehler = str(e)
    p = dec.progress()
    out = dict(gelesen=gelesen, frames=p.received, nutzbar=p.useful, bloecke=p.solved, K=p.K, fertig=p.complete)
    if fehler and not gelesen:
        out["fehler"] = fehler
    if p.complete:
        try:
            name, kind, data = dec.result()
            datei = data
            out.update(name=name, art=kind, bytes=len(data))
        except ValueError as e:
            out.update(fertig=False, fehler=f"Assembling: {e}")
    return out
