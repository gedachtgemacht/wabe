#!/usr/bin/env python3
"""
dicht.  (Arbeitsname)  --  hochdichter 2D-Farbcode als QR-Nachfolger, Prototyp v0.1

Kernideen
---------
* 8 Farben pro Modul (3 Bit statt 1 Bit) -> Ecken des RGB-Wuerfels, maximal unterscheidbar
* 4 Finder in den Ecken (einer davon anders geringt -> Orientierung) -> Homographie
* Farbpalette neben jedem Finder -> der Decoder kalibriert sich auf Druck, Display und Licht
  (bilinear ueber die Flaeche interpoliert, Lichtgradienten sind damit egal)
* Reed-Solomon ueber GF(256), Bloecke verschachtelt (Interleaving) -> Flecken/Kratzer verteilen sich
* Scrambler (xorshift32) statt QR-Masken -> keine grossen Einfarbflaechen, keine Fehl-Finder
* Header mit Dateityp, Dateiname, Chunk-Index -> mehrseitig auf Papier, mehrere Frames am Bildschirm
* Nutzdaten vorher mit Brotli (oder LZMA) komprimiert

Abhaengigkeiten: numpy, opencv-python, reedsolo, brotli (optional)
"""
from __future__ import annotations

import io
import math
import struct
import zlib
import lzma
import os
import sys
import argparse
from dataclasses import dataclass, field

import numpy as np
import cv2
from reedsolo import RSCodec, ReedSolomonError

try:
    import brotli
    HAVE_BROTLI = True
except ImportError:  # pragma: no cover
    HAVE_BROTLI = False

# --------------------------------------------------------------------------------------
# Konstanten
# --------------------------------------------------------------------------------------
MAGIC = b"dc"
VERSION = 1
FINDER = 9                     # Finder ist 9x9 Module
CORNER_ZONE = 12               # Finder + Trennring + Palette belegen max. 12 Zeilen in der Ecke
SIZES = [48, 64, 80, 100, 120, 150, 180, 220, 260, 320, 400, 500]   # Module pro Seite
DEFAULT_EC = 0.25              # Anteil Reed-Solomon-Redundanz

# Farbindex -> 3 Bit. Reihenfolge RGB.
PALETTE_RGB = np.array([
    [0, 0, 0],        # 0 schwarz
    [255, 255, 255],  # 1 weiss
    [255, 0, 0],      # 2 rot
    [0, 255, 0],      # 3 gruen
    [0, 0, 255],      # 4 blau
    [0, 255, 255],    # 5 cyan
    [255, 0, 255],    # 6 magenta
    [255, 255, 0],    # 7 gelb
], dtype=np.uint8)

TYPES = {"bin": 0, "txt": 1, "html": 2, "pdf": 3, "json": 4, "md": 5,
         "png": 6, "jpg": 7, "svg": 8, "csv": 9, "zip": 10}
TYPE_NAMES = {v: k for k, v in TYPES.items()}
COMP_NONE, COMP_BROTLI, COMP_LZMA = 0, 1, 2

# Header: magic(2) ver(1) flags(1) chunk_idx(2) chunk_total(2) total_len(4) crc32(4) payload_len(4) name_len(1)
HDR = struct.Struct("<2sBBHHIIIB")


# --------------------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------------------
def cell_map(S: int, k: int, r: int, c: int) -> tuple[int, int]:
    """Lokale Eckkoordinate (Zeile r, Spalte c) der Ecke k -> globale Zelle (Zeile, Spalte).
    Ecken: 0=oben links, 1=oben rechts, 2=unten rechts, 3=unten links."""
    gr = r if k in (0, 1) else S - 1 - r
    gc = c if k in (0, 3) else S - 1 - c
    return gr, gc


def point_map(S: int, k: int, y: float, x: float) -> tuple[float, float]:
    """Kontinuierliche Variante von cell_map fuer Punkte (y, x) in Moduleinheiten."""
    gy = y if k in (0, 1) else S - y
    gx = x if k in (0, 3) else S - x
    return gy, gx


def layout(S: int) -> tuple[np.ndarray, np.ndarray]:
    """Liefert (Festmuster als RGB-Bild SxSx3, Reserviert-Maske SxS)."""
    img = np.full((S, S, 3), 255, np.uint8)
    res = np.zeros((S, S), bool)
    for k in range(4):
        # Finder 9x9 plus 1 Modul weisser Trennring -> 10x10
        for r in range(FINDER + 1):
            for c in range(FINDER + 1):
                gr, gc = cell_map(S, k, r, c)
                res[gr, gc] = True
                if r < FINDER and c < FINDER:
                    d = min(r, c, FINDER - 1 - r, FINDER - 1 - c)
                    if k == 0:
                        black = d == 0 or d >= 3      # Ringe 1 schwarz, 2 weiss, Kern 3x3  (Orientierung)
                    else:
                        black = d == 0 or d >= 2      # Ringe 1 schwarz, 1 weiss, Kern 5x5
                    img[gr, gc] = 0 if black else 255
                else:
                    img[gr, gc] = 255
        # Palette: 2 Zeilen x 8 Farben direkt unter dem Finder
        for r in (FINDER + 1, FINDER + 2):
            for c in range(8):
                gr, gc = cell_map(S, k, r, c)
                res[gr, gc] = True
                img[gr, gc] = PALETTE_RGB[c]
    return img, res


_LAYOUT_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def get_layout(S: int):
    if S not in _LAYOUT_CACHE:
        _LAYOUT_CACHE[S] = layout(S)
    return _LAYOUT_CACHE[S]


# --------------------------------------------------------------------------------------
# Reed-Solomon, Interleaving, Scrambler
# --------------------------------------------------------------------------------------
_CODECS: dict[int, RSCodec] = {}


def get_codec(nsym: int) -> RSCodec:
    if nsym not in _CODECS:
        _CODECS[nsym] = RSCodec(nsym)
    return _CODECS[nsym]


def rs_plan(nbytes: int, ec: float) -> list[tuple[int, int]]:
    """Teilt nbytes Codewort-Bytes in Bloecke <= 255. Liefert [(n, nsym), ...]."""
    nb = max(1, math.ceil(nbytes / 255))
    base, extra = divmod(nbytes, nb)
    plan = []
    for i in range(nb):
        n = base + (1 if i < extra else 0)
        nsym = max(2, int(round(n * ec)))
        nsym = min(nsym, n - 1)
        plan.append((n, nsym))
    return plan


def capacity(S: int, ec: float = DEFAULT_EC) -> tuple[int, int, int]:
    """(Datenmodule, Codewort-Bytes, Nutzbytes nach Fehlerkorrektur)."""
    _, res = get_layout(S)
    M = int((~res).sum())
    B = (3 * M) // 8
    plan = rs_plan(B, ec)
    K = sum(n - ns for n, ns in plan)
    return M, B, K


def xorshift_mask(n: int, seed: int = 0x6D6B8A3F) -> np.ndarray:
    x = seed & 0xFFFFFFFF
    out = np.empty(n, np.uint8)
    for i in range(n):
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= x >> 17
        x ^= (x << 5) & 0xFFFFFFFF
        out[i] = x & 0xFF
    return out


_MASK_CACHE: dict[int, np.ndarray] = {}


def scramble(data: bytes) -> bytes:
    n = len(data)
    if n not in _MASK_CACHE:
        _MASK_CACHE[n] = xorshift_mask(n)
    return bytes(np.frombuffer(data, np.uint8) ^ _MASK_CACHE[n])


def interleave_order(plan):
    """Reihenfolge (block, pos) der Codewort-Bytes im Modulstrom."""
    maxn = max(n for n, _ in plan)
    order = []
    for p in range(maxn):
        for b, (n, _) in enumerate(plan):
            if p < n:
                order.append((b, p))
    return order


# --------------------------------------------------------------------------------------
# Encoder
# --------------------------------------------------------------------------------------
def encode_grid(S: int, data: bytes, ec: float = DEFAULT_EC) -> np.ndarray:
    """Packt `data` (<= capacity(S)[2] Bytes) in ein SxS-RGB-Modulbild."""
    fixed, res = get_layout(S)
    M = int((~res).sum())
    B = (3 * M) // 8
    plan = rs_plan(B, ec)
    K = sum(n - ns for n, ns in plan)
    if len(data) > K:
        raise ValueError(f"zu viele Daten: {len(data)} > {K} Bytes fuer S={S}")
    data = data + bytes(K - len(data))

    blocks, pos = [], 0
    for n, ns in plan:
        k = n - ns
        blocks.append(bytes(get_codec(ns).encode(data[pos:pos + k])))
        pos += k
    stream = bytearray(B)
    for t, (b, p) in enumerate(interleave_order(plan)):
        stream[t] = blocks[b][p]
    stream = scramble(bytes(stream))

    bits = np.unpackbits(np.frombuffer(stream, np.uint8))
    pad = 3 * M - len(bits)                       # 0..7 Fuellbits
    rng = np.random.default_rng(7)
    bits = np.concatenate([bits, rng.integers(0, 2, pad, dtype=np.uint8)])
    syms = bits.reshape(M, 3) @ np.array([4, 2, 1], np.uint8)

    grid = fixed.copy()
    ys, xs = np.nonzero(~res)                      # zeilenweise Leserichtung
    grid[ys, xs] = PALETTE_RGB[syms]
    return grid


def decode_grid(syms: np.ndarray, S: int, ec: float = DEFAULT_EC) -> tuple[bytes, int]:
    """Symbolbild SxS (Farbindizes 0..7) -> (Nutzbytes, korrigierte Fehler). Wirft ReedSolomonError."""
    _, res = get_layout(S)
    M = int((~res).sum())
    B = (3 * M) // 8
    plan = rs_plan(B, ec)
    s = syms[~res].astype(np.uint8)               # gleiche Leserichtung wie beim Encoder
    bits = ((s[:, None] >> np.array([2, 1, 0], np.uint8)) & 1).reshape(-1)
    stream = np.packbits(bits[: 8 * B]).tobytes()
    stream = scramble(stream)
    blocks = [bytearray(n) for n, _ in plan]
    for t, (b, p) in enumerate(interleave_order(plan)):
        blocks[b][p] = stream[t]
    out = bytearray()
    errs = 0
    for (n, ns), blk in zip(plan, blocks):
        msg, _, errpos = get_codec(ns).decode(bytes(blk))
        errs += len(errpos)
        out += msg
    return bytes(out), errs


# --------------------------------------------------------------------------------------
# Datei-Ebene: Kompression, Header, Chunks
# --------------------------------------------------------------------------------------
def compress_best(data: bytes) -> tuple[int, bytes]:
    cands = [(COMP_NONE, data)]
    if HAVE_BROTLI:
        cands.append((COMP_BROTLI, brotli.compress(data, quality=11, lgwin=24)))
    cands.append((COMP_LZMA, lzma.compress(data, preset=9 | lzma.PRESET_EXTREME)))
    return min(cands, key=lambda c: len(c[1]))


def decompress(comp: int, data: bytes) -> bytes:
    if comp == COMP_NONE:
        return data
    if comp == COMP_BROTLI:
        return brotli.decompress(data)
    if comp == COMP_LZMA:
        return lzma.decompress(data)
    raise ValueError("unbekannte Kompression")


def guess_type(name: str) -> str:
    ext = os.path.splitext(name)[1].lower().lstrip(".")
    ext = {"htm": "html", "jpeg": "jpg", "text": "txt"}.get(ext, ext)
    return ext if ext in TYPES else "bin"


def pick_size(nbytes: int, ec: float, min_size: int | None = None) -> int:
    for S in SIZES:
        if min_size and S < min_size:
            continue
        if capacity(S, ec)[2] >= nbytes:
            return S
    return SIZES[-1]


def encode_file(data: bytes, name: str = "", kind: str | None = None, S: int | None = None,
                ec: float = DEFAULT_EC) -> list[np.ndarray]:
    """Liefert eine Liste von Modulbildern (eine je Chunk/Seite/Frame)."""
    kind = kind or guess_type(name)
    name_b = name.encode("utf-8")[:255]
    comp, cdata = compress_best(data)
    crc = zlib.crc32(data) & 0xFFFFFFFF
    hdr_len = HDR.size + len(name_b)
    if S is None:
        S = pick_size(hdr_len + len(cdata), ec)
    cap = capacity(S, ec)[2] - hdr_len
    if cap <= 0:
        raise ValueError("Code zu klein fuer den Header")
    n_chunks = max(1, math.ceil(len(cdata) / cap))
    grids = []
    for i in range(n_chunks):
        chunk = cdata[i * cap:(i + 1) * cap]
        flags = (TYPES[kind] & 0xF) | (comp << 4)
        hdr = HDR.pack(MAGIC, VERSION, flags, i, n_chunks, len(cdata), crc, len(chunk), len(name_b))
        grids.append(encode_grid(S, hdr + name_b + chunk, ec))
    return grids


@dataclass
class Chunk:
    kind: str
    comp: int
    idx: int
    total: int
    total_len: int
    crc: int
    name: str
    payload: bytes


def parse_chunk(raw: bytes) -> Chunk:
    magic, ver, flags, idx, total, total_len, crc, plen, nlen = HDR.unpack(raw[:HDR.size])
    if magic != MAGIC or ver != VERSION:
        raise ValueError("kein dicht.-Code (Magic/Version falsch)")
    name = raw[HDR.size:HDR.size + nlen].decode("utf-8", "replace")
    payload = raw[HDR.size + nlen:HDR.size + nlen + plen]
    if len(payload) != plen:
        raise ValueError("Payload-Laenge passt nicht")
    return Chunk(TYPE_NAMES.get(flags & 0xF, "bin"), flags >> 4, idx, total, total_len, crc, name, payload)


class Collector:
    """Sammelt Chunks (mehrere Seiten / Frames) und setzt die Datei zusammen."""

    def __init__(self):
        self.chunks: dict[int, Chunk] = {}

    def add(self, ch: Chunk) -> bool:
        self.chunks[ch.idx] = ch
        return self.complete

    @property
    def total(self) -> int:
        return next(iter(self.chunks.values())).total if self.chunks else 0

    @property
    def complete(self) -> bool:
        return bool(self.chunks) and len(self.chunks) == self.total

    def result(self) -> tuple[str, str, bytes]:
        if not self.complete:
            raise ValueError(f"unvollstaendig: {len(self.chunks)}/{self.total}")
        first = self.chunks[0]
        cdata = b"".join(self.chunks[i].payload for i in range(first.total))
        if len(cdata) != first.total_len:
            raise ValueError("Gesamtlaenge passt nicht")
        data = decompress(first.comp, cdata)
        if (zlib.crc32(data) & 0xFFFFFFFF) != first.crc:
            raise ValueError("CRC-Fehler")
        return first.name, first.kind, data


# --------------------------------------------------------------------------------------
# Rendern
# --------------------------------------------------------------------------------------
def render(grid: np.ndarray, ppm: int = 8, quiet: int = 4) -> np.ndarray:
    """Modulbild -> RGB-Pixelbild mit `ppm` Pixel pro Modul und Ruhezone `quiet` Module."""
    S = grid.shape[0]
    big = np.full((S + 2 * quiet, S + 2 * quiet, 3), 255, np.uint8)
    big[quiet:quiet + S, quiet:quiet + S] = grid
    return np.repeat(np.repeat(big, ppm, axis=0), ppm, axis=1)


# --------------------------------------------------------------------------------------
# Decoder (Bild -> Daten)
# --------------------------------------------------------------------------------------
@dataclass
class DecodeInfo:
    S: int = 0
    ppm: float = 0.0
    rs_errors: int = 0
    finders: list = field(default_factory=list)
    H: np.ndarray | None = None
    note: str = ""


def _binarize(gray: np.ndarray) -> np.ndarray:
    """Beleuchtungsnormierte Schwarz/Weiss-Trennung. Schwarz -> 255."""
    h, w = gray.shape
    k = max(15, (min(h, w) // 25) | 1)
    bg = cv2.dilate(gray, np.ones((k, k), np.uint8))
    bg = cv2.blur(bg, (k, k)).astype(np.float32) + 1.0
    norm = gray.astype(np.float32) / bg
    return (norm < 0.55).astype(np.uint8) * 255


def _find_finders(binary: np.ndarray) -> list[dict]:
    cnts, hier = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return []
    hier = hier[0]
    cands = []
    for i, c in enumerate(cnts):
        a = cv2.contourArea(c)
        if a < 64:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.06 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        # Quadratisch genug? Seitenverhaeltnis ueber minAreaRect
        (_, (rw, rh), _) = cv2.minAreaRect(c)
        if min(rw, rh) < 0.55 * max(rw, rh):
            continue
        inner = None
        j = hier[i][2]
        while j >= 0:                      # Loecher
            g = hier[j][2]
            while g >= 0:                  # schwarze Kerne im Loch
                ag = cv2.contourArea(cnts[g])
                if ag > 0.04 * a and (inner is None or ag > inner[1]):
                    inner = (g, ag)
                g = hier[g][0]
            j = hier[j][0]
        if inner is None:
            continue
        m = cv2.moments(c)
        cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
        mg = cv2.moments(cnts[inner[0]])
        if mg["m00"] == 0:
            continue
        gx, gy = mg["m10"] / mg["m00"], mg["m01"] / mg["m00"]
        if math.hypot(cx - gx, cy - gy) > 0.12 * math.sqrt(a):
            continue
        ratio = inner[1] / a
        if ratio > 0.5:
            continue
        cands.append(dict(center=np.array([cx, cy]), corners=approx.reshape(4, 2).astype(np.float64),
                          area=a, ratio=ratio))
    return cands


def _select_finders(cands: list[dict]) -> list[dict]:
    special = [c for c in cands if c["ratio"] < 0.19]
    normal = [c for c in cands if c["ratio"] >= 0.19]
    if not special or len(normal) < 3:
        raise ValueError(f"Finder nicht gefunden ({len(special)} Orientierungs-, {len(normal)} Standard-Finder)")
    best = None
    for sp in special:
        # die 3 Standardfinder, die in der Groesse am besten zu sp passen
        nm = sorted(normal, key=lambda c: abs(math.log(c["area"] / sp["area"])))[:3]
        score = sum(abs(math.log(c["area"] / sp["area"])) for c in nm)
        if best is None or score < best[0]:
            best = (score, sp, nm)
    _, sp, nm = best
    allf = [sp] + nm
    ctr = np.mean([f["center"] for f in allf], axis=0)
    ang0 = math.atan2(*(sp["center"] - ctr)[::-1])
    def rel(f):
        a = math.atan2(*(f["center"] - ctr)[::-1]) - ang0
        return a % (2 * math.pi)
    nm.sort(key=rel)
    return [sp] + nm                       # TL, TR, BR, BL


def _finder_grid_corners(S: int, k: int) -> np.ndarray:
    """Globale Grid-Koordinaten (x, y) der 4 Aussenecken des Finders in Ecke k."""
    pts = []
    for (ly, lx) in [(0, 0), (0, FINDER), (FINDER, FINDER), (FINDER, 0)]:
        gy, gx = point_map(S, k, ly, lx)
        pts.append((gx, gy))
    return np.array(pts, np.float64)


def _finder_grid_center(S: int, k: int) -> np.ndarray:
    gy, gx = point_map(S, k, FINDER / 2, FINDER / 2)
    return np.array([gx, gy], np.float64)


def _estimate_S(finders: list[dict]) -> tuple[float, float]:
    ppm = np.mean([(math.sqrt(f["area"]) + 1.0) / FINDER for f in finders])
    c = [f["center"] for f in finders]
    d = np.mean([np.linalg.norm(c[i] - c[(i + 1) % 4]) for i in range(4)])
    return d / ppm + FINDER, ppm


def _homography(gray: np.ndarray, finders: list[dict], S: int, ppm: float) -> np.ndarray:
    src = np.array([_finder_grid_center(S, k) for k in range(4)], np.float32)
    dst = np.array([f["center"] for f in finders], np.float32)
    H0 = cv2.getPerspectiveTransform(src, dst)
    src_all, dst_all = [], []
    win = max(2, int(ppm * 0.5))
    for k, f in enumerate(finders):
        gc = _finder_grid_corners(S, k)
        proj = cv2.perspectiveTransform(gc.reshape(-1, 1, 2).astype(np.float32), H0).reshape(4, 2)
        used = set()
        for p, g in zip(proj, gc):
            d = np.linalg.norm(f["corners"] - p, axis=1)
            for idx in np.argsort(d):
                if idx not in used:
                    used.add(idx)
                    src_all.append(g)
                    dst_all.append(f["corners"][idx])
                    break
    dst_all = np.array(dst_all, np.float32).reshape(-1, 1, 2)
    try:
        dst_all = cv2.cornerSubPix(gray, dst_all, (win, win), (-1, -1),
                                   (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
    except cv2.error:
        pass
    src_all = np.array(src_all, np.float32)
    dst_all = dst_all.reshape(-1, 2)
    src_all = np.vstack([src_all, src])
    dst_all = np.vstack([dst_all, dst])
    H, _ = cv2.findHomography(src_all, dst_all, 0)
    return H


_OFFS = [(0.0, 0.0), (0.15, 0.15), (-0.15, 0.15), (0.15, -0.15), (-0.15, -0.15)]


def _sample_points(img: np.ndarray, H: np.ndarray, pts_xy: np.ndarray, offs=_OFFS) -> np.ndarray:
    """Mittelt Farbwerte um Grid-Punkte (x, y). img float32 BGR."""
    n = len(pts_xy)
    cols = 1024                                   # remap mag keine Kanten > 32767
    rows = math.ceil(n / cols)
    acc = np.zeros((rows * cols, 3), np.float32)
    for dx, dy in offs:
        p = (pts_xy + np.array([dx, dy])).reshape(-1, 1, 2).astype(np.float32)
        ip = cv2.perspectiveTransform(p, H).reshape(-1, 2)
        mx = np.zeros(rows * cols, np.float32); my = np.zeros(rows * cols, np.float32)
        mx[:n] = ip[:, 0]; my[:n] = ip[:, 1]
        acc += cv2.remap(img, mx.reshape(rows, cols), my.reshape(rows, cols), cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE).reshape(-1, 3)
    return acc[:n] / len(offs)


def _palette_refs(img: np.ndarray, H: np.ndarray, S: int) -> np.ndarray:
    refs = np.zeros((4, 8, 3), np.float32)
    offs = [(0.0, 0.0), (0.0, 0.45), (0.0, -0.45), (0.2, 0.2), (-0.2, -0.2)]
    for k in range(4):
        pts = []
        for j in range(8):
            gy, gx = point_map(S, k, FINDER + 2.0, j + 0.5)   # Mitte des 2x1-Blocks
            pts.append((gx, gy))
        refs[k] = _sample_points(img, H, np.array(pts), offs)
    return refs


def _verify(img: np.ndarray, H: np.ndarray, S: int, refs: np.ndarray) -> bool:
    """Trennring muss hell sein, Palette muss 8 verschiedene Farben zeigen."""
    for k in range(4):
        pts = []
        for c in range(FINDER + 1):
            gy, gx = point_map(S, k, FINDER + 0.5, c + 0.5)
            pts.append((gx, gy))
        for r in range(FINDER):
            gy, gx = point_map(S, k, r + 0.5, FINDER + 0.5)
            pts.append((gx, gy))
        sep = _sample_points(img, H, np.array(pts))
        white, black = refs[k, 1], refs[k, 0]
        contrast = np.linalg.norm(white - black)
        if contrast < 40:
            return False
        # Projektion auf schwarz->weiss-Achse
        t = ((sep - black) @ (white - black)) / (contrast ** 2)
        if np.mean(t) < 0.6:
            return False
        dists = [np.linalg.norm(refs[k, i] - refs[k, j]) for i in range(8) for j in range(i + 1, 8)]
        if min(dists) < 0.18 * contrast:
            return False
    return True


def _normalize_local(samples: np.ndarray, win: int = 15):
    """Lokale Schwarz-/Weisspegel je Kanal aus den Daten selbst (jeder Kanal ist 50/50 bimodal):
    zwei Iterationen 'Mittel der Werte unter/ueber der lokalen Schwelle'. Macht Vignette,
    Lichtgradient und Farbstich unschaedlich, ohne auf die Ecken angewiesen zu sein."""
    low = high = None
    for it in range(2):
        thr = cv2.blur(samples, (win, win)) if it == 0 else (low + high) / 2
        hi = (samples > thr).astype(np.float32)
        lo = 1.0 - hi
        high = cv2.blur(samples * hi, (win, win)) / (cv2.blur(hi, (win, win)) + 1e-3)
        low = cv2.blur(samples * lo, (win, win)) / (cv2.blur(lo, (win, win)) + 1e-3)
    return (samples - low) / np.maximum(high - low, 1e-3), low, high


def _palette_norm(refs: np.ndarray, low: np.ndarray, high: np.ndarray, S: int) -> np.ndarray:
    """Gemessene Eck-Paletten in den normierten Raum bringen und mitteln -> 8x3 Referenzen."""
    rn = []
    for k in range(4):
        cells = [cell_map(S, k, r, c) for r in (FINDER + 1, FINDER + 2) for c in range(8)]
        ys = [c[0] for c in cells]
        xs = [c[1] for c in cells]
        l = low[ys, xs].mean(0)
        h = high[ys, xs].mean(0)
        rn.append((refs[k] - l) / np.maximum(h - l, 1e-3))
    return np.mean(rn, 0).astype(np.float32)


def _nearest(x: np.ndarray, refs: np.ndarray) -> np.ndarray:
    d = ((x[:, :, None, :] - refs) ** 2).sum(-1)
    return d.argmin(-1).astype(np.uint8)


def _equalize(nrm: np.ndarray, cls: np.ndarray, refs: np.ndarray, iters: int = 2):
    """Entzerrt das Uebersprechen benachbarter Module (Unschaerfe, Chroma-Subsampling):
    schaetzt einen 3x3-Kern (Mitte/Kante/Ecke) per kleinster Quadrate aus der aktuellen
    Klassifikation und zieht den Nachbaranteil ab. 2 Iterationen reichen."""
    coef = None
    for _ in range(iters):
        I = refs[cls]
        P = np.pad(I, ((1, 1), (1, 1), (0, 0)), mode="edge")
        E = P[:-2, 1:-1] + P[2:, 1:-1] + P[1:-1, :-2] + P[1:-1, 2:]
        C = P[:-2, :-2] + P[:-2, 2:] + P[2:, :-2] + P[2:, 2:]
        A = np.stack([I.reshape(-1), E.reshape(-1), C.reshape(-1), np.ones(I.size, np.float32)], 1)
        coef, *_ = np.linalg.lstsq(A, nrm.reshape(-1), rcond=None)
        a, b, c, d0 = coef
        if a < 0.3:
            break
        x = (nrm - b * E - c * C - d0) / a
        cls = _nearest(x, refs)
    return cls, coef


def _classify(samples: np.ndarray, refs: np.ndarray, S: int) -> np.ndarray:
    """samples SxSx3 (BGR), refs 4x8x3 gemessene Eck-Paletten -> Farbindizes SxS."""
    nrm, low, high = _normalize_local(samples)
    pn = _palette_norm(refs, low, high, S)
    cls = _nearest(nrm, pn)
    cls, _ = _equalize(nrm, cls, pn)
    return cls


def decode_image(img_bgr: np.ndarray, ec: float = DEFAULT_EC, debug: bool = False):
    """Foto/Scan/Screenshot (BGR uint8) -> (Chunk, DecodeInfo). Wirft ValueError/ReedSolomonError."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    binary = _binarize(gray)
    finders = _select_finders(_find_finders(binary))
    S_est, ppm = _estimate_S(finders)
    order = sorted(SIZES, key=lambda s: abs(math.log(s / S_est)))[:3]
    imgf = img_bgr.astype(np.float32)
    info = DecodeInfo(ppm=ppm, finders=finders)
    verify_err, rs_err = None, None
    for S in order:
        H = _homography(gray, finders, S, ppm)
        refs = _palette_refs(imgf, H, S)
        if not _verify(imgf, H, S, refs):
            verify_err = verify_err or ValueError(f"Rastergroesse unklar (geschaetzt {S_est:.0f})")
            continue
        ys, xs = np.mgrid[0:S, 0:S]
        pts = np.stack([xs + 0.5, ys + 0.5], -1).reshape(-1, 2).astype(np.float64)
        samples = _sample_points(imgf, H, pts).reshape(S, S, 3)
        syms = _classify(samples, refs, S)
        try:
            raw, errs = decode_grid(syms, S, ec)
            chunk = parse_chunk(raw)
        except (ReedSolomonError, ValueError) as e:
            rs_err = ValueError(f"S={S}: zu viele Fehler fuer Reed-Solomon ({e})")
            continue
        info.S, info.H, info.rs_errors = S, H, errs
        info.note = f"S={S} ppm={ppm:.1f} RS-Korrekturen={errs}"
        return chunk, info
    raise rs_err or verify_err or ValueError("Dekodierung fehlgeschlagen")


def draw_debug(img_bgr: np.ndarray, info: DecodeInfo) -> np.ndarray:
    out = img_bgr.copy()
    for k, f in enumerate(info.finders):
        cv2.polylines(out, [f["corners"].astype(np.int32).reshape(-1, 1, 2)], True, (0, 0, 255), 2)
        cv2.putText(out, "TL TR BR BL".split()[k], tuple(f["center"].astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    if info.H is not None and info.S:
        S = info.S
        box = np.array([[0, 0], [S, 0], [S, S], [0, S]], np.float32).reshape(-1, 1, 2)
        ip = cv2.perspectiveTransform(box, info.H).astype(np.int32)
        cv2.polylines(out, [ip], True, (0, 200, 0), 2)
    return out


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="dicht. Prototyp")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("encode", help="Datei -> PNG(s)")
    e.add_argument("infile")
    e.add_argument("-o", "--out", default="dicht")
    e.add_argument("-s", "--size", type=int, default=None, help="Module pro Seite (sonst automatisch)")
    e.add_argument("--ppm", type=int, default=8)
    e.add_argument("--ec", type=float, default=DEFAULT_EC)
    d = sub.add_parser("decode", help="Bild(er) -> Datei")
    d.add_argument("images", nargs="+")
    d.add_argument("-o", "--out", default=None)
    c = sub.add_parser("capacity", help="Kapazitaetstabelle")
    c.add_argument("--ec", type=float, default=DEFAULT_EC)
    a = ap.parse_args(argv)

    if a.cmd == "capacity":
        print(f"EC-Anteil {a.ec:.0%}")
        print(f"{'Module':>7} {'Datenmod.':>10} {'Nutzbytes':>10}  Vergleich QR v40 = 2953 B")
        for S in SIZES:
            M, B, K = capacity(S, a.ec)
            print(f"{S:>4}x{S:<3} {M:>10} {K:>10}  ({K / 2953:.1f}x)")
    elif a.cmd == "encode":
        data = open(a.infile, "rb").read()
        grids = encode_file(data, os.path.basename(a.infile), S=a.size, ec=a.ec)
        for i, g in enumerate(grids):
            fn = f"{a.out}.png" if len(grids) == 1 else f"{a.out}_{i + 1:03d}.png"
            cv2.imwrite(fn, cv2.cvtColor(render(g, a.ppm), cv2.COLOR_RGB2BGR))
            print(fn, f"S={g.shape[0]}")
        print(f"{len(data)} Bytes in {len(grids)} Code(s)")
    elif a.cmd == "decode":
        col = Collector()
        for fn in a.images:
            img = cv2.imread(fn)
            chunk, info = decode_image(img)
            print(f"{fn}: Chunk {chunk.idx + 1}/{chunk.total} {info.note}")
            col.add(chunk)
        name, kind, data = col.result()
        out = a.out or (name or f"dicht_out.{kind}")
        open(out, "wb").write(data)
        print(f"-> {out} ({len(data)} Bytes, {kind})")


if __name__ == "__main__":
    main()
