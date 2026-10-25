"""Minimaler FastAPI-Endpunkt fuer wabe. Strom: Das Handy schickt alle 300 bis 500 ms ein
JPEG (auf ~1500 px verkleinert, Kreis moeglichst bildfuellend), der Server sammelt die Frames
und meldet den Fortschritt. Sobald fertig, liefert /strom/<sid>/datei die Datei.

    pip install fastapi uvicorn python-multipart
    uvicorn server_beispiel:app --host 0.0.0.0 --port 8000
"""
import io
import time
import uuid

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response

import strom
import wabe

app = FastAPI()
SESSIONS: dict[str, dict] = {}        # sid -> {"dec": StreamDecoder, "t": letzter Zugriff}


def _session(sid: str):
    s = SESSIONS.get(sid)
    if s is None:
        s = SESSIONS[sid] = {"dec": strom.StreamDecoder(), "t": time.time()}
    s["t"] = time.time()
    # alte Sitzungen aufraeumen
    for k in [k for k, v in SESSIONS.items() if time.time() - v["t"] > 600]:
        SESSIONS.pop(k, None)
    return s["dec"]


@app.post("/strom/neu")
def neu():
    return {"sid": uuid.uuid4().hex}


@app.post("/strom/{sid}/frame")
async def frame(sid: str, file: UploadFile = File(...)):
    dec = _session(sid)
    buf = np.frombuffer(await file.read(), np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        return {"ok": False, "fehler": "kein Bild"}
    t = time.time()
    try:
        p = dec.add_image(img)
    except Exception as e:                          # Frame unlesbar: einfach weiter
        p = dec.progress()
        return {"ok": False, "fehler": str(e)[:120], "fortschritt": p.note, "fertig": p.complete,
                "dauer_s": round(time.time() - t, 2)}
    return {"ok": True, "fortschritt": p.note, "bloecke": p.solved, "K": p.K, "frames": p.received,
            "fertig": p.complete, "dauer_s": round(time.time() - t, 2)}


@app.get("/strom/{sid}/datei")
def datei(sid: str):
    dec = _session(sid)
    name, kind, data = dec.result()                 # wirft, solange unvollstaendig
    media = {"pdf": "application/pdf", "html": "text/html", "txt": "text/plain", "json": "application/json",
             "png": "image/png", "jpg": "image/jpeg"}.get(kind, "application/octet-stream")
    return Response(data, media_type=media, headers={"Content-Disposition": f'inline; filename="{name}"'})


@app.post("/einzel")
async def einzel(file: UploadFile = File(...)):
    """Einzelner Code (Papier oder Bildschirmblatt): Datei direkt zurueck."""
    buf = np.frombuffer(await file.read(), np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    col = wabe.dicht.Collector()
    for chunk, info in wabe.decode_all(img):
        col.add(chunk)
    if not col.complete:
        return {"ok": False, "fehler": f"Chunks {len(col.chunks)}/{col.total}"}
    name, kind, data = col.result()
    return Response(data, media_type="application/octet-stream",
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})
