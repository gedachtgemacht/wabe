#!/usr/bin/env python3
"""wabe. Kamera-Test: nimmt Fotos entgegen, dekodiert sie mit wabe.decode_image, sammelt Seiten je Sitzung.

POST /api/decode   Body = Bilddatei (JPEG/PNG), Header X-Sitzung
GET  /api/datei    Header/Query sitzung -> fertige Datei
POST /api/neu      Sitzung leeren
POST /api/strom    Body = Videoframe (JPEG), Header X-Sitzung: Strom-Frames sammeln (strom.py), Fortschritt zurueck;
                   ist alles da, liegt die Datei unter /api/datei
"""
import base64
import collections
import json
import multiprocessing as mp
import os
import struct
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, quote, urlparse

import cv2
import numpy as np

import dicht
import strom
import wabe

PORT = int(os.environ.get("WABE_PORT", "8915"))
MAX_BODY = 25 * 1024 * 1024
MAX_LONG_EDGE = 6000
TIMEOUT_S = 90
STROM_TIMEOUT_S = 20
SESSION_TTL = 3600
MAX_SESSIONS = 200

# Strom-Diagnose: Logzeilen (auch ins Journal) und die letzten fehlgeschlagenen Frames im RAM
strom_log: collections.deque = collections.deque(maxlen=500)
strom_bilder: collections.deque = collections.deque(maxlen=15)     # (laufende Nr, Zeit, Fehler, JPEG-Bytes)
strom_nr = 0

slots = threading.BoundedSemaphore(4)
sessions: dict[str, dict] = {}
lock = threading.Lock()


def _work(conn, raw: bytes):
    try:
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Bild nicht lesbar")
        h, w = img.shape[:2]
        s = MAX_LONG_EDGE / max(h, w)
        if s < 1:
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        t = time.time()
        out = {"bild": [int(img.shape[1]), int(img.shape[0])]}
        try:
            chunk, info = wabe.decode_image(img)
            out.update(ok=True, chunk=chunk, note=info.note, rs=info.rs_errors)
        except Exception as e:  # Decoder wirft ValueError, ReedSolomonError, cv2.error ...
            out.update(ok=False, fehler=str(e) or type(e).__name__)
            info = None
        out["sek"] = round(time.time() - t, 1)
        vis = wabe.draw_debug(img, info) if info is not None else img
        vs = 900 / max(vis.shape[:2])
        if vs < 1:
            vis = cv2.resize(vis, None, fx=vs, fy=vs, interpolation=cv2.INTER_AREA)
        out["debug"] = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes()
        conn.send(out)
    except Exception as e:
        conn.send({"ok": False, "fehler": f"{type(e).__name__}: {e}"})
    finally:
        conn.close()


def _work_strom(conn, raw: bytes):
    """Videoframe -> rohe Nutzbytes aller lesbaren Codes darin (das Loesen macht der Hauptprozess)."""
    try:
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Bild nicht lesbar")
        h, w = img.shape[:2]
        bild = [int(w), int(h)]
        s = 3200 / max(h, w)
        s_srv = min(1.0, s)
        if s < 1:
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        t = time.time()
        try:
            found = wabe.decode_all(img, parse=False)
            payloads = [bytes(raw_) for raw_, _ in found]
            # Kamerapixel je Zelle im gesendeten Bild (die Seite regelt ihre Bildgroesse danach)
            ppm = round(max(i.ppm / i.scale for _, i in found) / s_srv, 2)
            rs = max(i.rs_errors for _, i in found)               # korrigierte Bytes: Mass fuer die Reserve
            conn.send({"ok": True, "payloads": payloads, "sek": round(time.time() - t, 2), "bild": bild, "ppm": ppm, "rs": rs})
        except Exception as e:
            conn.send({"ok": False, "fehler": str(e) or type(e).__name__, "sek": round(time.time() - t, 2),
                       "bild": bild})
    except Exception as e:
        conn.send({"ok": False, "fehler": f"{type(e).__name__}: {e}"})
    finally:
        conn.close()


def decode_isoliert(raw: bytes, ziel=_work, limit=TIMEOUT_S) -> dict:
    """Eigener Prozess je Foto, damit ein haengender Decoder den Server nicht blockiert."""
    rx, tx = mp.Pipe(duplex=False)
    p = mp.Process(target=ziel, args=(tx, raw), daemon=True)
    p.start()
    tx.close()
    try:
        if rx.poll(limit):
            return rx.recv()
        return {"ok": False, "fehler": f"Zeitlimit ({limit} s)"}
    except EOFError:
        return {"ok": False, "fehler": "Decoder abgestuerzt"}
    finally:
        if p.is_alive():
            p.kill()
        p.join()


def sitzung(sid: str) -> dict:
    now = time.time()
    with lock:
        for k in [k for k, v in sessions.items() if now - v["t"] > SESSION_TTL]:
            del sessions[k]
        if sid not in sessions:
            if len(sessions) >= MAX_SESSIONS:
                del sessions[min(sessions, key=lambda k: sessions[k]["t"])]
            sessions[sid] = {"col": dicht.Collector(), "t": now, "strom": strom.StreamDecoder(),
                             "strom_lock": threading.Lock()}
        sessions[sid]["t"] = now
        return sessions[sid]


def antwort_decode(sid: str, raw: bytes) -> dict:
    if not slots.acquire(blocking=False):
        return {"ok": False, "fehler": "Server ausgelastet, gleich nochmal"}
    try:
        res = decode_isoliert(raw)
    finally:
        slots.release()
    out = {k: v for k, v in res.items() if k not in ("chunk", "debug")}
    if res.get("debug"):
        out["debug"] = "data:image/jpeg;base64," + base64.b64encode(res["debug"]).decode()
    ses = sitzung(sid)
    chunk = res.get("chunk")
    if chunk is not None:
        col = ses["col"]
        if col.chunks and col.total != chunk.total:
            ses["col"] = col = dicht.Collector()
        col.add(chunk)
        out.update(seite=chunk.idx + 1, von=chunk.total, da=sorted(i + 1 for i in col.chunks),
                   name=chunk.name, art=chunk.kind)
        if col.complete:
            try:
                name, kind, data = col.result()
                ses["datei"] = (name, kind, data)
                out.update(fertig=True, name=name, art=kind, bytes=len(data),
                           link=f"api/datei?sitzung={quote(sid)}")
            except ValueError as e:
                out.update(fertig=False, fehler=f"Zusammensetzen: {e}")
    return out


def strom_protokoll(sid: str, raw: bytes, res: dict, out: dict, kamera: str = "", schaerfe: str = "", zeiten: str = ""):
    global strom_nr
    with lock:
        strom_nr += 1
        nr = strom_nr
        zeit = datetime.now().strftime("%H:%M:%S")
        zeile = (f"{zeit} #{nr} {sid[:6]} {len(raw) // 1024} KB kamera={kamera} schaerfe={schaerfe} vorher[{zeiten}] ppm={res.get('ppm')} rs={res.get('rs')} bild={res.get('bild')} sek={res.get('sek')} "
                 f"codes={len(res.get('payloads', []))} " +
                 ("ok" if out["ok"] else f"FEHLER: {out.get('fehler')}") +
                 f" | {out['note']}" + (" FERTIG" if out["fertig"] else ""))
        strom_log.append(zeile)
        if not out["ok"]:
            strom_bilder.append((nr, zeit, out.get("fehler"), raw))
    print(zeile, flush=True)


def antwort_strom(sid: str, raw: bytes, kamera: str = "", schaerfe: str = "", zeiten: str = "") -> dict:
    if not slots.acquire(blocking=False):
        with lock:
            strom_log.append(f"{datetime.now():%H:%M:%S} {sid[:6]} ausgelastet")
        return {"ok": False, "fehler": "Server ausgelastet", "busy": True}
    try:
        res = decode_isoliert(raw, _work_strom, STROM_TIMEOUT_S)
    finally:
        slots.release()
    ses = sitzung(sid)
    with ses["strom_lock"]:
        dec = ses["strom"]
        fehler = res.get("fehler")
        gelesen = 0
        for pl in res.get("payloads", []):
            try:
                dec.add_payload(pl)
                gelesen += 1
            except (ValueError, struct.error) as e:      # lesbarer Code, aber kein Strom-Frame
                fehler = str(e)
        p = dec.progress()
        out = {"ok": gelesen > 0, "sek": res.get("sek"), "frames": p.received, "nutzbar": p.useful,
               "bloecke": p.solved, "K": p.K, "note": p.note, "fertig": p.complete, "ppm": res.get("ppm")}
        if not gelesen:
            out["fehler"] = fehler or "kein Code"
        if p.complete:
            try:
                name, kind, data = dec.result()
                ses["datei"] = (name, kind, data)
                out.update(name=name, art=kind, bytes=len(data), link=f"api/datei?sitzung={quote(sid)}")
            except ValueError as e:
                out.update(fertig=False, fehler=f"Zusammensetzen: {e}")
        strom_protokoll(sid, raw, res, out, kamera, schaerfe, zeiten)
        return out


MIME = {"pdf": "application/pdf", "html": "text/html; charset=utf-8", "txt": "text/plain; charset=utf-8",
        "png": "image/png", "jpg": "image/jpeg", "json": "application/json"}


class H(BaseHTTPRequestHandler):
    server_version = "wabe"

    def log_message(self, fmt, *a):
        pass

    def _json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def _sid(self):
        q = parse_qs(urlparse(self.path).query)
        sid = self.headers.get("X-Sitzung") or (q.get("sitzung") or [""])[0]
        return sid if 8 <= len(sid) <= 64 and sid.isalnum() else None

    def do_POST(self):
        path = urlparse(self.path).path
        sid = self._sid()
        if not sid:
            return self._json({"ok": False, "fehler": "Sitzung fehlt"}, 400)
        if path == "/api/neu":
            with lock:
                sessions.pop(sid, None)
            return self._json({"ok": True})
        if path == "/api/lokal/log":                      # Messzeilen aus dem Lokal-Modus (Decoder im Browser), auch gesammelt
            n = min(int(self.headers.get("Content-Length") or 0), 60000)
            for zeile in self.rfile.read(n).decode("utf-8", "replace").split("\n")[-300:]:
                zeile = f"{datetime.now():%H:%M:%S} LOKAL {sid[:6]} {zeile[:400]}"
                with lock:
                    strom_log.append(zeile)
                print(zeile, flush=True)
            return self._json({"ok": True})
        if path not in ("/api/decode", "/api/strom"):
            return self._json({"ok": False}, 404)
        n = int(self.headers.get("Content-Length") or 0)
        if not 0 < n <= MAX_BODY:
            return self._json({"ok": False, "fehler": "Bild zu gross oder leer"}, 413)
        raw = self.rfile.read(n)
        if path == "/api/strom":
            return self._json(antwort_strom(sid, raw, (self.headers.get("X-Kamera") or "")[:20],
                                                        (self.headers.get("X-Schaerfe") or "")[:12],
                                                        (self.headers.get("X-Zeiten") or "")[:60]))
        self._json(antwort_decode(sid, raw))

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/health":
            return self._json({"ok": True, "sitzungen": len(sessions)})
        if path == "/api/strom/log":
            b = "\n".join(strom_log).encode() + b"\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            return self.wfile.write(b)
        if path == "/api/strom/bilder":
            q = parse_qs(urlparse(self.path).query)
            nr = (q.get("nr") or [""])[0]
            if not nr:
                return self._json([{"nr": n, "zeit": z, "fehler": f, "bytes": len(r)} for n, z, f, r in strom_bilder])
            for n, z, f, r in strom_bilder:
                if str(n) == nr:
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(r)))
                    self.end_headers()
                    return self.wfile.write(r)
            return self._json({"ok": False}, 404)
        if path != "/api/datei":
            return self._json({"ok": False}, 404)
        sid = self._sid()
        ses = sessions.get(sid) if sid else None
        if not ses or "datei" not in ses:
            return self._json({"ok": False, "fehler": "keine Datei"}, 404)
        name, kind, data = ses["datei"]
        fn = name or f"wabe.{kind}"
        self.send_response(200)
        self.send_header("Content-Type", MIME.get(kind, "application/octet-stream"))
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Content-Disposition", f"inline; filename*=UTF-8''{quote(fn)}")
        self.send_header("Content-Security-Policy", "sandbox")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)


if __name__ == "__main__":
    mp.set_start_method("fork")
    print(f"wabe server auf 127.0.0.1:{PORT}")
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
