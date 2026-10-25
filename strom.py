"""
wabe. Strom  --  Datei vom Bildschirm aufs Handy durch Draufhalten (Prototyp)

Idee: Der Bildschirm zeigt eine Folge von wabe.-Codes (Frames). Jeder Frame traegt eine
zufaellige XOR-Kombination der Dateibloecke (Random Linear Fountain ueber GF(2)). Das Handy
muss weder den ersten Frame erwischen noch jeden: beliebige K+2 Frames reichen fast immer,
egal in welcher Reihenfolge, egal wie viele verloren gehen. Kein Sync, kein Rueckkanal.

Frame-Inhalt (innerhalb der Reed-Solomon-geschuetzten Nutzdaten eines wabe.-Codes):
  magic 'ws', ver, flags(typ|komp<<4), stream_id(4), K(2), block_len(2), total_len(4), crc32(4),
  seed(4), name_len(1), name, dann block_len Bytes XOR-Kombination.
"""
from __future__ import annotations

import math
import struct
import zlib
import base64
import io
from dataclasses import dataclass, field

import numpy as np
import cv2

import wabe
import dicht

SHDR = struct.Struct("<2sBBIHHIIIB")
SMAGIC = b"ws"
SVER = 1


def _mask_from_seed(seed: int, K: int) -> np.ndarray:
    """Zufaellige Teilmenge der K Bloecke (jeder mit Wahrscheinlichkeit 1/2, nie leer)."""
    rng = np.random.default_rng(seed)
    while True:
        m = rng.integers(0, 2, K, dtype=np.uint8)
        if m.any():
            return m


class StreamEncoder:
    def __init__(self, data: bytes, name: str, R: int = 128, shape: str = "circle",
                 ec_idx: int = 1, levels: int = 2, stream_id: int | None = None, payload_cap: int | None = None,
                 color: bool = False):
        self.kind = dicht.guess_type(name)
        self.name_b = name.encode("utf-8")[:40]
        self.comp, self.cdata = dicht.compress_best(data)
        self.crc = zlib.crc32(data) & 0xFFFFFFFF
        self.R, self.shape, self.ec_idx, self.levels, self.color = R, shape, ec_idx, levels, color
        cap = payload_cap or wabe.capacity(R, shape, wabe.EC_LEVELS[ec_idx], levels, color)[2]
        self.block_len = cap - SHDR.size - len(self.name_b)
        if self.block_len < 64:
            raise ValueError("Code zu klein")
        self.K = max(1, math.ceil(len(self.cdata) / self.block_len))
        padded = self.cdata + bytes(self.K * self.block_len - len(self.cdata))
        self.blocks = np.frombuffer(padded, np.uint8).reshape(self.K, self.block_len)
        self.stream_id = stream_id if stream_id is not None else int(np.random.default_rng().integers(1, 2**32))
        self._n = 0

    def frame_payload(self, seed: int) -> bytes:
        mask = _mask_from_seed(seed, self.K)
        block = np.bitwise_xor.reduce(self.blocks[mask.astype(bool)], axis=0)
        flags = (dicht.TYPES[self.kind] & 0xF) | (self.comp << 4)
        hdr = SHDR.pack(SMAGIC, SVER, flags, self.stream_id, self.K, self.block_len, len(self.cdata),
                        self.crc, seed, len(self.name_b))
        return hdr + self.name_b + block.tobytes()

    def frame_cells(self, seed: int) -> np.ndarray:
        return wabe.encode_cells(self.R, self.shape, self.frame_payload(seed), self.ec_idx, self.levels, self.color)

    def next_frame(self) -> np.ndarray:
        """Naechster Frame als Zellwerte; Seeds laufen einfach hoch (jeder Seed = eine Kombination)."""
        self._n += 1
        return self.frame_cells(self._n)

    def frames(self, n: int) -> list[np.ndarray]:
        return [self.next_frame() for _ in range(n)]


@dataclass
class StreamProgress:
    received: int = 0
    useful: int = 0
    solved: int = 0
    K: int = 0
    complete: bool = False
    note: str = ""


class StreamDecoder:
    """Sammelt Frames (Zellwerte oder Fotos) und loest das Gleichungssystem ueber GF(2) inkrementell."""

    def __init__(self):
        self.meta = None
        self.rows: list[np.ndarray] = []        # reduzierte Zeilen (Masken) in Zeilenstufenform
        self.vals: list[np.ndarray] = []        # zugehoerige Bytevektoren
        self.pivots: list[int] = []
        self.received = 0
        self.useful = 0

    def add_payload(self, raw: bytes) -> StreamProgress:
        magic, ver, flags, sid, K, blen, total, crc, seed, nlen = SHDR.unpack(raw[:SHDR.size])
        if magic != SMAGIC or ver != SVER:
            raise ValueError("kein wabe.-Strom")
        name = raw[SHDR.size:SHDR.size + nlen].decode("utf-8", "replace")
        block = np.frombuffer(raw[SHDR.size + nlen:SHDR.size + nlen + blen], np.uint8).copy()
        if len(block) != blen:
            raise ValueError("Block unvollstaendig")
        if self.meta is None or self.meta["sid"] != sid:
            self.meta = dict(sid=sid, K=K, blen=blen, total=total, crc=crc, name=name,
                             kind=dicht.TYPE_NAMES.get(flags & 0xF, "bin"), comp=flags >> 4)
            self.rows, self.vals, self.pivots = [], [], []
            self.received = self.useful = 0
        self.received += 1
        mask = _mask_from_seed(seed, K)
        # Gauss-Elimination: neue Zeile gegen vorhandene Pivots reduzieren
        for r, v, p in zip(self.rows, self.vals, self.pivots):
            if mask[p]:
                mask = mask ^ r
                block = block ^ v
        if mask.any():
            p = int(np.argmax(mask))
            # vorhandene Zeilen mit Bit p ebenfalls reduzieren (reduzierte Stufenform)
            for i in range(len(self.rows)):
                if self.rows[i][p]:
                    self.rows[i] = self.rows[i] ^ mask
                    self.vals[i] = self.vals[i] ^ block
            self.rows.append(mask)
            self.vals.append(block)
            self.pivots.append(p)
            self.useful += 1
        return self.progress()

    def add_cells(self, cells: np.ndarray, R: int, shape: str = "circle", ec_idx: int = 1, levels: int = 2):
        raw, _ = wabe.decode_cells(cells, R, shape, ec_idx, levels)
        return self.add_payload(raw)

    def add_image(self, img) -> StreamProgress:
        """Foto/Videoframe -> alle lesbaren Codes darin einsammeln."""
        results = wabe.decode_all(img, parse=False)
        prog = None
        for raw, info in results:
            prog = self.add_payload(raw)
        return prog or self.progress()

    def progress(self) -> StreamProgress:
        K = self.meta["K"] if self.meta else 0
        return StreamProgress(self.received, self.useful, len(self.pivots), K, bool(K) and len(self.pivots) == K,
                              f"{len(self.pivots)}/{K} Bloecke aus {self.received} Frames")

    def result(self) -> tuple[str, str, bytes]:
        if not self.progress().complete:
            raise ValueError("unvollstaendig: " + self.progress().note)
        K, blen = self.meta["K"], self.meta["blen"]
        blocks = np.zeros((K, blen), np.uint8)
        for r, v, p in zip(self.rows, self.vals, self.pivots):
            blocks[p] = v                           # reduzierte Stufenform: jede Zeile = genau ein Block
        cdata = blocks.tobytes()[: self.meta["total"]]
        data = dicht.decompress(self.meta["comp"], cdata)
        if (zlib.crc32(data) & 0xFFFFFFFF) != self.meta["crc"]:
            raise ValueError("CRC-Fehler")
        return self.meta["name"], self.meta["kind"], data


# -------------------------------------------------------------------- Ausgabe fuer den Bildschirm
def player_html(frames_png: list[bytes], fps: float = 4.0, title: str = "wabe. Strom") -> str:
    """Eigenstaendige HTML-Seite, die die Frames pixelgenau (Geraetepixel 1:1) in Schleife zeigt."""
    imgs = ",".join('"data:image/png;base64,' + base64.b64encode(p).decode() + '"' for p in frames_png)
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>
<style>html,body{{margin:0;background:#fff;height:100%;overflow:hidden}}
canvas{{display:block;margin:0 auto;image-rendering:pixelated}}
#info{{position:fixed;left:8px;bottom:6px;font:12px system-ui;color:#999}}</style></head><body>
<canvas id="c"></canvas><div id="info"></div>
<script>
const srcs=[{imgs}];const fps={fps};const imgs=[];let loaded=0;
const c=document.getElementById('c'),ctx=c.getContext('2d');
function fit(){{const dpr=window.devicePixelRatio||1;const w=imgs[0].width,h=imgs[0].height;
c.width=w;c.height=h;c.style.width=(w/dpr)+'px';c.style.height=(h/dpr)+'px';
c.style.marginTop=Math.max(0,(window.innerHeight-h/dpr)/2)+'px';}}
let order=[],p=0;   // jede Runde in neuer Zufallsreihenfolge: kein fester Takt, der mit der Kamera in Gleichtakt geraet
function mischen(){{order=imgs.map((_,k)=>k);for(let k=order.length-1;k>0;k--){{const j=Math.floor(Math.random()*(k+1));[order[k],order[j]]=[order[j],order[k]];}}p=0;}}
function tick(){{if(p>=order.length)mischen();const i=order[p++];ctx.drawImage(imgs[i],0,0);document.getElementById('info').textContent=
'Frame '+(i+1)+'/'+imgs.length+' · '+fps+' fps · 1 Bildpixel = 1 Geraetepixel';}}
srcs.forEach((s,k)=>{{const im=new Image();im.onload=()=>{{loaded++;if(loaded===srcs.length){{fit();mischen();tick();
setInterval(tick,1000/fps);window.addEventListener('resize',fit);}}}};im.src=s;imgs[k]=im;}});
</script></body></html>"""


def frame_png(cells: np.ndarray, R: int, shape: str = "circle", px: float = 3.5, quiet: float = 3.0,
              color: bool = False) -> bytes:
    img = wabe.render(cells, R, shape, px, quiet, color=color)
    if color:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes()
