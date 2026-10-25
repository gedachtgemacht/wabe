#!/usr/bin/env python3
"""
wabe.  (working title)  --  round black/white code on a hexagonal grid, prototype v0.1

Structure
---------
* Cells on a hexagonal grid (distance between neighbouring cell centres = 1 "pitch").
  Compared with squares: 15 % more cells at the same minimum spacing (= same camera resolution).
* Outline: circle (default) or hexagon, radius R in pitch.
* Finder: bullseye in the centre (3 black rings), rotation-invariant and extremely robust to detect.
* Markers: small ring discs (2 rings) on rings around the centre, 6 per ring, outermost at R-5,
  then every 40 pitch inwards. They provide perspective, orientation and a displacement field
  against lens distortion / wavy paper.
* Format ring directly around the bullseye: 16 bits (version, shape, size, EC) with RS(6,2), twice.
* Data: Reed-Solomon GF(256), interleaved blocks, xorshift scrambler, Brotli, file header
  with type, name and chunk index (taken over from dicht.py).
* Decoder: illumination-normalised, contour detection, homography from centre + 6 outer markers,
  rotation via the format ring (6 attempts), thin-plate spline over all markers,
  local black/white levels from the data, equalisation of neighbour crosstalk (6 neighbours).
"""
from __future__ import annotations

import io
import math
import os
import sys
import struct
import argparse
from dataclasses import dataclass, field

import numpy as np
import cv2
from reedsolo import RSCodec, ReedSolomonError

import dicht  # container, RS plan, scrambler, interleaving (lives in the same folder)

# --------------------------------------------------------------------------------------
RADII = [32, 40, 50, 64, 80, 100, 128, 160, 200, 256, 320, 400, 500]         # radius in pitch
SHAPES = {"circle": 0, "hexagon": 1, "rect": 2}       # rect: aspect ratio 16:9, R = half the height
RECT_ASPECT = 16.0 / 9.0
SHAPE_NAMES = {v: k for k, v in SHAPES.items()}
EC_LEVELS = [0.15, 0.25, 0.35, 0.50]
FORMAT_VERSION = 1

BULLSEYE = [(2.0, 1), (3.5, 0), (4.75, 1), (6.25, 0), (7.5, 1), (9.0, 0), (10.25, 1), (11.75, 0)]   # (radius up to, ink)
BULL_OUTER = 10.25                                  # outer edge of the last black ring
FORMAT_R = (11.75, 13.75)
MARKER = [(1.5, 1), (2.75, 0), (4.0, 1), (5.0, 0)]                        # alignment marker (1 ring)
PRINCIPAL = [(1.5, 1), (2.75, 0), (4.0, 1), (5.25, 0), (6.5, 1), (7.75, 0)]  # 6 principal markers on the outside (2 rings)
PRINCIPAL_OUTER, MARKER_OUTER = 6.5, 4.0
MARKER_STEP = 40
MARKER_MARGIN = 8
SQ3 = math.sqrt(3.0)


# --------------------------------------------------------------------------------------
# Grid
# --------------------------------------------------------------------------------------
def axial_to_xy(q, r):
    return q + r / 2.0, r * SQ3 / 2.0


def xy_to_axial_round(x, y):
    """Continuous (x, y) -> nearest cell (q, r), vectorised (cube rounding)."""
    q = x - y / SQ3
    r = 2.0 * y / SQ3
    s = -q - r
    rq, rr, rs = np.round(q), np.round(r), np.round(s)
    dq, dr, ds = np.abs(rq - q), np.abs(rr - r), np.abs(rs - s)
    m1 = (dq > dr) & (dq > ds)
    m2 = (~m1) & (dr > ds)
    rq = np.where(m1, -rr - rs, rq)
    rr = np.where(m2, -rq - rs, rr)
    return rq.astype(np.int32), rr.astype(np.int32)


@dataclass
class Lattice:
    R: int
    shape: str
    q: np.ndarray            # (N,) axial
    r: np.ndarray
    xy: np.ndarray           # (N,2) cell centres in pitch
    dist: np.ndarray         # (N,) distance to the centre
    table: np.ndarray        # dense table (q+off, r+off) -> index or -1
    off: int
    fixed: np.ndarray        # (N,) -1 data, 0 fixed white, 1 fixed black
    data_idx: np.ndarray     # indices of the data cells in reading order
    format_idx: np.ndarray   # indices of the format cells (sorted by angle)
    markers: list = field(default_factory=list)   # [(ring_k, j, xy)] grid coordinates of the markers
    neighbors: np.ndarray | None = None           # (N,6) neighbour indices or -1

    @property
    def n_data_bits(self):
        return len(self.data_idx)


_LAT_CACHE: dict[tuple, Lattice] = {}


def marker_positions(R: int, shape: str = "circle") -> list[tuple[int, int, tuple[float, float]]]:
    """Marker centres in grid coordinates: ring k, marker j, (x, y).
    Ring 0 = 6 principal markers on the outside at rho0 = R-8 (perspective + orientation, 2 rings, larger).
    Ring 1 = alignment markers on the same radius between the principal markers, then every 40 pitch
    inwards, dense enough (about every 40 pitch along the ring) for lens and paper waves."""
    out = []
    rho0 = (R * SQ3 / 2 if shape == "hexagon" else R) - MARKER_MARGIN
    if rho0 < 20:
        return out
    for j in range(6):
        a = math.radians(60 * j)
        out.append((0, j, (rho0 * math.cos(a), rho0 * math.sin(a))))
    rho, k = rho0, 1
    while rho >= 24:
        m = max(1, int(math.floor(2 * math.pi * rho / (6 * MARKER_STEP) + 0.5)))
        n = 6 * m
        for j in range(n):
            if k == 1 and j % m == 0:
                continue                      # a principal marker sits there
            a = math.radians(360.0 * j / n + (180.0 / n if k % 2 == 0 else 0.0))
            out.append((k, j, (rho * math.cos(a), rho * math.sin(a))))
        rho -= MARKER_STEP
        k += 1
    if shape == "rect":
        # Area outside the inner circle: alignment markers on a grid every 40 pitch,
        # ring index by distance band (the decoder works its way from the inside outwards)
        W = R * RECT_ASPECT
        have = np.array([m_[2] for m_ in out])
        j = 0
        for gx in np.arange(-W + MARKER_MARGIN, W - MARKER_MARGIN + 1e-9, MARKER_STEP):
            for gy in np.arange(-R + MARKER_MARGIN, R - MARKER_MARGIN + 1e-9, MARKER_STEP):
                d = math.hypot(gx, gy)
                if d <= rho0 + 12 or (len(have) and np.min(np.hypot(have[:, 0] - gx, have[:, 1] - gy)) < 14):
                    continue
                out.append((100 + int((d - rho0) // MARKER_STEP), j, (float(gx), float(gy))))   # k>=100: grid marker
                j += 1
    return out


def build_lattice(R: int, shape: str = "circle") -> Lattice:
    key = (R, shape)
    if key in _LAT_CACHE:
        return _LAT_CACHE[key]
    lim = int(R * (RECT_ASPECT + 0.6 if shape == "rect" else 1.2)) + 3
    qq, rr = np.meshgrid(np.arange(-lim, lim + 1), np.arange(-lim, lim + 1), indexing="ij")
    qq, rr = qq.ravel(), rr.ravel()
    x, y = axial_to_xy(qq.astype(np.float64), rr.astype(np.float64))
    if shape == "circle":
        inside = np.hypot(x, y) <= R + 1e-9
    elif shape == "hexagon":
        inside = np.maximum(np.maximum(np.abs(qq), np.abs(rr)), np.abs(qq + rr)) <= R
    elif shape == "rect":
        inside = (np.abs(x) <= R * RECT_ASPECT + 1e-9) & (np.abs(y) <= R + 1e-9)
    else:
        raise ValueError("unknown shape")
    q, r, x, y = qq[inside], rr[inside], x[inside], y[inside]
    N = len(q)
    xy = np.stack([x, y], 1)
    dist = np.hypot(x, y)
    off = lim
    table = np.full((2 * lim + 1, 2 * lim + 1), -1, np.int32)
    table[q + off, r + off] = np.arange(N)

    fixed = np.full(N, -1, np.int8)
    # Bullseye
    prev = -1.0
    for rad, ink in BULLSEYE:
        m = (dist > prev) & (dist <= rad)
        fixed[m] = ink
        prev = rad
    # Format ring
    fm = (dist > FORMAT_R[0]) & (dist <= FORMAT_R[1])
    ang = np.arctan2(y, x)
    fidx = np.nonzero(fm)[0]
    fidx = fidx[np.argsort(ang[fidx], kind="stable")]
    fixed[fm] = 0
    # Marker
    markers = []
    for (mk, mj, (mx, my)) in marker_positions(R, shape):
        d = np.hypot(x - mx, y - my)
        prev = -1.0
        pat = PRINCIPAL if mk == 0 else MARKER
        for rad, ink in pat:
            m = (d > prev) & (d <= rad)
            fixed[m] = ink
            prev = rad
        # control point = ink centroid of the marker cells (grid quantisation shifts
        # the centre by up to 0.5 pitch relative to the target position)
        ink_cells = (d <= pat[-2][0]) & (fixed == 1)
        cx, cy = x[ink_cells].mean(), y[ink_cells].mean()
        markers.append((mk, mj, (float(cx), float(cy))))
    data_mask = fixed < 0
    didx = np.nonzero(data_mask)[0]
    ring = np.round(dist[didx]).astype(np.int64)
    order = np.lexsort((ang[didx], ring))
    didx = didx[order]
    # Neighbours
    nb = np.full((N, 6), -1, np.int32)
    for i, (dq, dr) in enumerate([(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]):
        nq, nr = q + dq + off, r + dr + off
        ok = (nq >= 0) & (nq < table.shape[0]) & (nr >= 0) & (nr < table.shape[1])
        nb[ok, i] = table[nq[ok], nr[ok]]
    lat = Lattice(R, shape, q, r, xy, dist, table, off, fixed, didx, fidx, markers, nb)
    _LAT_CACHE[key] = lat
    return lat


# --------------------------------------------------------------------------------------
# Format info
# --------------------------------------------------------------------------------------
_FMT_CODEC = RSCodec(3)


def _crc8(data: bytes) -> int:
    c = 0
    for b in data:
        c ^= b
        for _ in range(8):
            c = ((c << 1) ^ 0x07) & 0xFF if c & 0x80 else (c << 1) & 0xFF
    return c


def pack_format(shape: str, R: int, ec_idx: int, levels: int = 2, color: bool = False) -> np.ndarray:
    v = ((FORMAT_VERSION & 3) | (SHAPES[shape] << 2) | (RADII.index(R) << 5) | (ec_idx << 10)
         | ((1 if levels == 4 else 0) << 12) | ((1 if color else 0) << 13))
    msg = struct.pack("<H", v)
    msg += bytes([_crc8(msg)])
    cw = bytes(_FMT_CODEC.encode(msg))                            # 6 bytes = 48 bits
    return np.unpackbits(np.frombuffer(cw, np.uint8))


def unpack_format(bits: np.ndarray):
    cw = np.packbits(bits[:48]).tobytes()
    msg, _, _ = _FMT_CODEC.decode(cw)
    msg = bytes(msg)
    if _crc8(msg[:2]) != msg[2]:
        raise ValueError("format CRC")
    v = struct.unpack("<H", msg[:2])[0]
    ver, shape, ridx, ec, lv, col = v & 3, (v >> 2) & 7, (v >> 5) & 31, (v >> 10) & 3, (v >> 12) & 1, (v >> 13) & 1
    if ver != FORMAT_VERSION or shape not in SHAPE_NAMES or ridx >= len(RADII):
        raise ValueError("format info invalid")
    return SHAPE_NAMES[shape], RADII[ridx], ec, (4 if lv else 2), bool(col)


# --------------------------------------------------------------------------------------
# Capacity and encoder
# --------------------------------------------------------------------------------------
def bits_per_cell(levels: int = 2, color: bool = False) -> int:
    return (2 if levels == 4 else 1) * (3 if color else 1)


def capacity(R: int, shape: str = "circle", ec: float = 0.25, levels: int = 2, color: bool = False):
    lat = build_lattice(R, shape)
    nbits = lat.n_data_bits * bits_per_cell(levels, color)
    B = nbits // 8
    plan = dicht.rs_plan(B, ec)
    K = sum(n - ns for n, ns in plan)
    return len(lat.q), nbits, K


def encode_cells(R: int, shape: str, data: bytes, ec_idx: int = 1, levels: int = 2,
                 color: bool = False) -> np.ndarray:
    """-> cell values (N,). Black/white: ink level 0 (white) .. levels-1 (black).
    Colour: one ink bit per channel, value = R<<2 | G<<1 | B (0 = white, 7 = black)."""
    lat = build_lattice(R, shape)
    ec = EC_LEVELS[ec_idx]
    bpc = bits_per_cell(levels, color)
    nbits = lat.n_data_bits * bpc
    B = nbits // 8
    plan = dicht.rs_plan(B, ec)
    K = sum(n - ns for n, ns in plan)
    if len(data) > K:
        raise ValueError(f"too much data: {len(data)} > {K}")
    data = data + bytes(K - len(data))
    blocks, pos = [], 0
    for n, ns in plan:
        k = n - ns
        blocks.append(bytes(dicht.get_codec(ns).encode(data[pos:pos + k])))
        pos += k
    stream = bytearray(B)
    for t, (b, p) in enumerate(dicht.interleave_order(plan)):
        stream[t] = blocks[b][p]
    stream = dicht.scramble(bytes(stream))
    bits = np.unpackbits(np.frombuffer(stream, np.uint8))
    pad = nbits - len(bits)
    rng = np.random.default_rng(11)
    bits = np.concatenate([bits, rng.integers(0, 2, pad, dtype=np.uint8)])
    black = 7 if color else levels - 1
    cells = np.where(lat.fixed < 0, 0, lat.fixed * black).astype(np.uint8)
    if color:
        cells[lat.data_idx] = bits[0::3] * 4 + bits[1::3] * 2 + bits[2::3]
    elif bpc == 1:
        cells[lat.data_idx] = bits
    else:
        cells[lat.data_idx] = bits[0::2] * 2 + bits[1::2]
    fbits = pack_format(shape, R, ec_idx, levels, color)
    fi = lat.format_idx
    reps = len(fi) // 48
    for rep in range(reps):
        cells[fi[rep * 48:(rep + 1) * 48]] = fbits * black
    return cells


def decode_cells(cells: np.ndarray, R: int, shape: str, ec_idx: int, levels: int = 2,
                 color: bool = False) -> tuple[bytes, int]:
    lat = build_lattice(R, shape)
    ec = EC_LEVELS[ec_idx]
    bpc = bits_per_cell(levels, color)
    nbits = lat.n_data_bits * bpc
    B = nbits // 8
    plan = dicht.rs_plan(B, ec)
    sym = cells[lat.data_idx].astype(np.uint8)
    if color:
        bits = np.stack([(sym >> 2) & 1, (sym >> 1) & 1, sym & 1], 1).reshape(-1)
    elif bpc == 1:
        bits = sym
    else:
        bits = np.stack([sym >> 1, sym & 1], 1).reshape(-1)
    stream = np.packbits(bits[:8 * B]).tobytes()
    stream = dicht.scramble(stream)
    blocks = [bytearray(n) for n, _ in plan]
    for t, (b, p) in enumerate(dicht.interleave_order(plan)):
        blocks[b][p] = stream[t]
    out = bytearray()
    errs = 0
    synds = _syndromes(blocks, plan)
    for (n, ns), blk, synd in zip(plan, blocks, synds):
        if not synd.any():
            out += bytes(blk[: n - ns])
            continue
        msg, npos = _rs_correct(blk, n, ns, synd)
        errs += npos
        out += msg
    return bytes(out), errs


def _rs_correct(blk: bytearray, n: int, ns: int, synd: np.ndarray):
    """Reed-Solomon correction of a block with syndromes that were already computed (vectorised);
    Berlekamp-Massey, Chien and Forney come from reedsolo (only for faulty blocks now)."""
    import reedsolo as rs
    dicht.get_codec(ns)                                   # makes sure the GF tables are initialised
    msg_out = bytearray(blk)
    s_list = [0] + [int(v) for v in synd]
    fsynd = rs.rs_forney_syndromes(s_list, [], n, 2)
    err_loc = rs.rs_find_error_locator(fsynd, ns, erase_count=0)
    err_pos = rs.rs_find_errors(err_loc[::-1], n, 2)
    if err_pos is None:
        raise ReedSolomonError("Could not locate error")
    msg_out = rs.rs_correct_errata(msg_out, s_list, err_pos, 0, 2)
    if _syndromes([msg_out], [(n, ns)])[0].any():
        raise ReedSolomonError("Could not correct message")
    return bytes(msg_out[: n - ns]), len(err_pos)


_GF_EXP = np.zeros(512, np.int32)
_GF_LOG = np.zeros(256, np.int32)
_x = 1
for _i in range(255):
    _GF_EXP[_i] = _x
    _GF_LOG[_x] = _i
    _x <<= 1
    if _x & 0x100:
        _x ^= 0x11D
_GF_EXP[255:510] = _GF_EXP[:255]


def _syndromes(blocks, plan) -> list[np.ndarray]:
    """Reed-Solomon syndromes per block, vectorised over GF(256) (generator 2, primitive polynomial 0x11D,
    fcr=0, msg[0] = highest coefficient, like reedsolo)."""
    res = []
    for (n, ns), blk in zip(plan, blocks):
        c = np.frombuffer(bytes(blk), np.uint8).astype(np.int32)
        nz = c != 0
        if not nz.any():
            res.append(np.zeros(ns, np.int32))
            continue
        deg = (n - 1 - np.arange(n))[nz]
        logs = _GF_LOG[c[nz]]
        j = np.arange(ns)[:, None]
        terms = _GF_EXP[(logs[None, :] + (j * deg[None, :]) % 255) % 255]
        res.append(np.bitwise_xor.reduce(terms, axis=1))
    return res


def _syndromes_zero(blocks, plan) -> list[bool]:
    return [not sd.any() for sd in _syndromes(blocks, plan)]


def pick_radius(nbytes: int, shape: str, ec: float, levels: int = 2, color: bool = False) -> int:
    for R in RADII:
        if capacity(R, shape, ec, levels, color)[2] >= nbytes:
            return R
    return RADII[-1]


def encode_file(data: bytes, name: str = "", kind: str | None = None, R: int | None = None,
                shape: str = "circle", ec_idx: int = 1, levels: int = 2, color: bool = False) -> list[np.ndarray]:
    """File -> list of cell value arrays (one per chunk). levels=4: 2 bits per cell (grey levels),
    color=True: 3 channels per cell (screen)."""
    kind = kind or dicht.guess_type(name)
    name_b = name.encode("utf-8")[:255]
    comp, cdata = dicht.compress_best(data)
    crc = __import__("zlib").crc32(data) & 0xFFFFFFFF
    hdr_len = dicht.HDR.size + len(name_b)
    ec = EC_LEVELS[ec_idx]
    if R is None:
        R = pick_radius(hdr_len + len(cdata), shape, ec, levels, color)
    cap = capacity(R, shape, ec, levels, color)[2] - hdr_len
    if cap <= 0:
        raise ValueError("code too small for the header")
    n_chunks = max(1, math.ceil(len(cdata) / cap))
    out = []
    for i in range(n_chunks):
        chunk = cdata[i * cap:(i + 1) * cap]
        flags = (dicht.TYPES[kind] & 0xF) | (comp << 4)
        hdr = dicht.HDR.pack(dicht.MAGIC, dicht.VERSION, flags, i, n_chunks, len(cdata), crc, len(chunk), len(name_b))
        out.append(encode_cells(R, shape, hdr + name_b + chunk, ec_idx, levels, color))
    return out


# --------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------
def render(cells: np.ndarray, R: int, shape: str = "circle", ppm: float = 6.0, quiet: float = 3.0,
           style: str = "hex", color: bool = False, levels: int = 2) -> np.ndarray:
    """Cell values -> image (uint8). Black/white or grey levels: single channel. color=True: RGB.
    style 'hex' = full honeycomb cells, 'dot' = dots (black/white only)."""
    lat = build_lattice(R, shape)
    ext_x = (R * RECT_ASPECT if shape == "rect" else R) + quiet
    ext_y = R + quiet
    w, h = int(math.ceil(2 * ext_x * ppm)), int(math.ceil(2 * ext_y * ppm))
    i, j = np.mgrid[0:h, 0:w]
    x = (j + 0.5) / ppm - ext_x
    y = (i + 0.5) / ppm - ext_y
    q, r = xy_to_axial_round(x, y)
    qi, ri = q + lat.off, r + lat.off
    ok = (qi >= 0) & (qi < lat.table.shape[0]) & (ri >= 0) & (ri < lat.table.shape[1])
    idx = np.full(x.shape, -1, np.int32)
    idx[ok] = lat.table[qi[ok], ri[ok]]
    valid = idx >= 0
    vals = cells[idx[valid]].astype(np.int32)
    if color:
        img = np.full((h, w, 3), 255, np.uint8)
        rgb = np.stack([255 * (1 - ((vals >> 2) & 1)), 255 * (1 - ((vals >> 1) & 1)), 255 * (1 - (vals & 1))], 1)
        img[valid] = rgb.astype(np.uint8)
        return img
    img = np.full((h, w), 255, np.uint8)
    lv = levels - 1
    if lv > 1:
        img[valid] = (255 - vals * 255 // lv).astype(np.uint8)
        return img
    black = np.zeros_like(valid)
    black[valid] = vals > 0
    if style == "dot":
        cx, cy = axial_to_xy(q.astype(np.float64), r.astype(np.float64))
        black &= np.hypot(x - cx, y - cy) <= 0.47
    img[black] = 0
    return img


def screen_sheets(data: bytes, name: str, width: int = 1920, height: int = 1080, px: float = 3.5,
                  levels: int = 4, cols: int = 2, margin: int = 40, gap: int = 40, quiet: float = 3.0):
    """File -> list of screen sheets (grey images width x height), up to cols x rows codes
    per sheet. Display 1:1 (no scaling!), then photograph: one photo per sheet."""
    tile_w = (width - 2 * margin - (cols - 1) * gap) / cols
    tile_h = height - 2 * margin
    R_max = min(tile_w, tile_h) / (2 * px) - quiet
    R = max([r for r in RADII if r <= R_max] or [RADII[0]])
    tile = int(round(2 * (R + quiet) * px))
    rows = max(1, int((height - 2 * margin + gap) // (tile + gap)))
    chunks = encode_file(data, name, R=R, levels=levels)
    per_sheet = cols * rows
    sheets = []
    for s0 in range(0, len(chunks), per_sheet):
        sheet = np.full((height, width), 255, np.uint8)
        group = chunks[s0:s0 + per_sheet]
        n_cols = min(cols, len(group))
        n_rows = math.ceil(len(group) / cols)
        x_start = (width - n_cols * tile - (n_cols - 1) * gap) // 2
        y_start = (height - n_rows * tile - (n_rows - 1) * gap) // 2
        for i, cl in enumerate(group):
            img = render(cl, R, "circle", px, quiet, "hex")
            img = cv2.resize(img, (tile, tile), interpolation=cv2.INTER_AREA) if img.shape[0] != tile else img
            r, c = divmod(i, cols)
            y = y_start + r * (tile + gap)
            x = x_start + c * (tile + gap)
            sheet[y:y + tile, x:x + tile] = img
        sheets.append(sheet)
    return sheets, R, len(chunks)


# --------------------------------------------------------------------------------------
# Decoder
# --------------------------------------------------------------------------------------
@dataclass
class DecodeInfo:
    R: int = 0
    shape: str = ""
    ppm: float = 0.0
    rs_errors: int = 0
    rotation: int = 0
    levels: int = 2
    color: bool = False
    scale: float = 1.0                 # working image / original image (centre and markers in working-image pixels)
    center: np.ndarray | None = None
    markers: list = field(default_factory=list)
    H: np.ndarray | None = None
    note: str = ""


def _normalize_illum(gray: np.ndarray, k: int):
    """Local black/white levels from the image itself (two iterations), -> float in ~[0,1].
    The level maps are smooth, so they are computed at 1/4 size."""
    g = gray.astype(np.float32)
    f = 4
    gs = cv2.resize(g, (max(1, g.shape[1] // f), max(1, g.shape[0] // f)), interpolation=cv2.INTER_AREA)
    ks = max(3, (k // f) | 1)
    low = high = None
    for it in range(2):
        thr = cv2.blur(gs, (ks, ks)) if it == 0 else (low + high) / 2
        hi = (gs > thr).astype(np.float32)
        lo = 1.0 - hi
        high = cv2.blur(gs * hi, (ks, ks)) / (cv2.blur(hi, (ks, ks)) + 1e-3)
        low = cv2.blur(gs * lo, (ks, ks)) / (cv2.blur(lo, (ks, ks)) + 1e-3)
    low = cv2.resize(low, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_LINEAR)
    high = cv2.resize(high, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_LINEAR)
    return (g - low) / np.maximum(high - low, 12.0)


def _binarize(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    k = max(15, (min(h, w) // 25) | 1)
    bg = cv2.dilate(gray, np.ones((k, k), np.uint8))
    bg = cv2.blur(bg, (k, k)).astype(np.float32) + 1.0
    return ((gray.astype(np.float32) / bg) < 0.55).astype(np.uint8) * 255


def _find_rings(binary: np.ndarray):
    """Contour tree: objects with a hole and a black core inside. Returns candidates with depth."""
    cnts, hier = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return []
    hier = hier[0]
    areas = [cv2.contourArea(c) for c in cnts]

    def core_depth(i):
        """Number of nested black cores below contour i (across holes)."""
        best = 0
        j = hier[i][2]
        while j >= 0:
            g = hier[j][2]
            while g >= 0:
                if areas[g] > 0.02 * areas[i]:
                    best = max(best, 1 + core_depth(g))
                g = hier[g][0]
            j = hier[j][0]
        return best

    # nesting depth: even levels are outer edges of black areas, odd levels are holes
    level = np.zeros(len(cnts), np.int32)
    for i in range(len(cnts)):
        p, lv = hier[i][3], 0
        while p >= 0:
            lv += 1
            p = hier[p][3]
        level[i] = lv

    cands = []
    for i, c in enumerate(cnts):
        a = areas[i]
        if a < 30 or hier[i][2] < 0 or level[i] % 2 == 1:
            continue
        hull = cv2.convexHull(c)
        ha, hp = cv2.contourArea(hull), cv2.arcLength(hull, True)
        circ = 4 * math.pi * ha / (hp * hp + 1e-9)       # the hull smooths the honeycomb edge
        if circ < 0.6 or a < 0.6 * ha:
            continue
        depth = core_depth(i)
        if depth < 1:
            continue
        m = cv2.moments(c)
        cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
        # the innermost core must be centred
        j = hier[i][2]
        inner = None
        while j >= 0:
            g = hier[j][2]
            while g >= 0:
                if areas[g] > 0.02 * a and (inner is None or areas[g] > areas[inner]):
                    inner = g
                g = hier[g][0]
            j = hier[j][0]
        mi = cv2.moments(cnts[inner])
        if mi["m00"] <= 0:
            continue
        ix, iy = mi["m10"] / mi["m00"], mi["m01"] / mi["m00"]
        if math.hypot(cx - ix, cy - iy) > 0.2 * math.sqrt(a):
            continue
        cands.append(dict(center=np.array([cx, cy]), area=a, depth=depth, r_px=math.sqrt(a / math.pi),
                          contour=c))
    return cands


def _subpixel_center(gray_n: np.ndarray, c: np.ndarray, r_px: float) -> np.ndarray:
    """Refine the centre of a ring disc: centroid of the ink in the circumscribed circle."""
    x0, y0 = int(round(c[0])), int(round(c[1]))
    rad = int(round(r_px * 1.05)) + 1          # only the ring disc itself (symmetric), no data cells
    h, w = gray_n.shape
    xs, xe, ys, ye = max(0, x0 - rad), min(w, x0 + rad + 1), max(0, y0 - rad), min(h, y0 + rad + 1)
    patch = 1.0 - np.clip(gray_n[ys:ye, xs:xe], 0, 1)
    yy, xx = np.mgrid[ys:ye, xs:xe]
    mask = (np.hypot(xx - c[0], yy - c[1]) <= rad).astype(np.float32)
    wgt = patch * mask
    s = wgt.sum()
    if s < 1e-3:
        return c
    return np.array([(wgt * xx).sum() / s, (wgt * yy).sum() / s])


def _tps_fit(src: np.ndarray, dst: np.ndarray, lam: float = 0.0):
    """Thin-plate spline src(N,2)->dst(N,2). Returns parameters for _tps_eval."""
    n = len(src)
    d = np.linalg.norm(src[:, None, :] - src[None, :, :], axis=-1)
    K = np.where(d > 0, d ** 2 * np.log(d + 1e-12), 0.0) + lam * np.eye(n)
    P = np.hstack([np.ones((n, 1)), src])
    A = np.zeros((n + 3, n + 3))
    A[:n, :n] = K
    A[:n, n:] = P
    A[n:, :n] = P.T
    b = np.zeros((n + 3, 2))
    b[:n] = dst
    sol = np.linalg.lstsq(A, b, rcond=None)[0]
    return src, sol[:n], sol[n:]


def _tps_eval_exact(params, pts: np.ndarray, chunk: int = 200000) -> np.ndarray:
    src, w, a = params
    out = np.empty_like(pts, dtype=np.float64)
    for s in range(0, len(pts), chunk):
        p = pts[s:s + chunk]
        d = np.linalg.norm(p[:, None, :] - src[None, :, :], axis=-1)
        U = np.where(d > 0, d ** 2 * np.log(d + 1e-12), 0.0)
        out[s:s + chunk] = U @ w + a[0] + p @ a[1:]
    return out


def _tps_eval(params, pts: np.ndarray, step: float = 4.0) -> np.ndarray:
    """Thin-plate spline at many points: the field is smooth, so it is evaluated exactly only on a
    grid with spacing `step` pitch, bilinearly in between (16x fewer kernel evaluations)."""
    if len(pts) < 4000:
        return _tps_eval_exact(params, pts)
    x0, y0 = pts.min(0) - step
    x1, y1 = pts.max(0) + step
    nx, ny = int((x1 - x0) / step) + 2, int((y1 - y0) / step) + 2
    gx, gy = np.meshgrid(x0 + np.arange(nx) * step, y0 + np.arange(ny) * step)
    field = _tps_eval_exact(params, np.stack([gx.ravel(), gy.ravel()], 1)).reshape(ny, nx, 2).astype(np.float32)
    n = len(pts)
    cols = 1024                                      # remap does not accept edges > 32767
    rows = int(math.ceil(n / cols))
    mx = np.zeros(rows * cols, np.float32)
    my = np.zeros(rows * cols, np.float32)
    mx[:n] = (pts[:, 0] - x0) / step
    my[:n] = (pts[:, 1] - y0) / step
    out = cv2.remap(field, mx.reshape(rows, cols), my.reshape(rows, cols), cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE)
    return out.reshape(-1, 2)[:n].astype(np.float64)


def _sample(img: np.ndarray, pts_img: np.ndarray, offs_px: np.ndarray) -> np.ndarray:
    """Mean of a single-channel float image around image points (N,2), offsets in pixels (M,2)."""
    n = len(pts_img)
    cols = 1024
    rows = math.ceil(n / cols)
    acc = np.zeros(rows * cols, np.float32)
    for o in offs_px:
        mx = np.zeros(rows * cols, np.float32)
        my = np.zeros(rows * cols, np.float32)
        mx[:n] = pts_img[:, 0] + o[0]
        my[:n] = pts_img[:, 1] + o[1]
        acc += cv2.remap(img, mx.reshape(rows, cols), my.reshape(rows, cols), cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE).ravel()
    return acc[:n] / len(offs_px)


def _rot(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])


def _equalize(v: np.ndarray, lat: Lattice, iters: int = 2, levels: int = 2):
    """v: normalised brightness per cell (0 black .. 1 white, from the 50/50 normalisation).
    Equalises 6-neighbour crosstalk and quantises to `levels` ink levels.
    With 4 levels the levels after the two-class normalisation are at -0.25, 0.25, 0.75, 1.25."""
    L = levels - 1
    if levels == 4:
        v = (v + 0.25) / 1.5                                  # -> 0 .. 1
    ink = np.clip(np.round((1.0 - v) * L), 0, L) / L          # ink 0..1 in steps
    nb = lat.neighbors
    coef = None
    for _ in range(iters):
        sp = np.append(ink, 0.0)                              # missing neighbours = paper
        nsum = sp[np.where(nb < 0, len(ink), nb)].sum(1)
        A = np.stack([ink, nsum, np.ones_like(ink)], 1)
        coef, *_ = np.linalg.lstsq(A, v, rcond=None)
        a, b, c = coef
        if a > -0.2:
            break
        corrected = (v - b * nsum - c) / a                    # ~ ink estimate in [0,1]
        ink = np.clip(np.round(corrected * L), 0, L) / L
    return np.round(ink * L).astype(np.uint8), coef


def _refine_grid(imgf: np.ndarray, lat: Lattice, to_img, rot: int, tile: float = 16.0,
                 eps: float = 0.25):
    """Data-driven fine alignment: for each tile (tile x tile pitch) the sub-pitch shift is searched
    at which the sampled cell values are most bimodal (mean distance from 0.5).
    Corrects residual errors from lens, paper waves and marker quantisation without further markers.
    Returns the image points of all cells after correction."""
    base = to_img(lat.xy, rot)
    # local Jacobian of the mapping grid -> image via finite differences (homography only,
    # the derivative of the displacement field is negligible by comparison)
    b0 = to_img(lat.xy, rot, tps_on=False)
    jx = (to_img(lat.xy + np.array([eps, 0.0]), rot, tps_on=False) - b0) / eps
    jy = (to_img(lat.xy + np.array([0.0, eps]), rot, tps_on=False) - b0) / eps
    R = lat.R
    nt = int(math.ceil(2 * R / tile))
    tx = np.clip(((lat.xy[:, 0] + R) / tile).astype(int), 0, nt - 1)
    ty = np.clip(((lat.xy[:, 1] + R) / tile).astype(int), 0, nt - 1)
    tid = ty * nt + tx
    count = np.bincount(tid, minlength=nt * nt).astype(np.float32)
    center = np.array([[0.0, 0.0]], np.float32)

    def score(dxc, dyc):
        """dxc, dyc: shift per cell (N,) in pitch -> bimodality per tile."""
        p = base + jx * dxc[:, None] + jy * dyc[:, None]
        v = _sample(imgf, p, center)
        return np.bincount(tid, weights=np.abs(v - 0.5), minlength=nt * nt) / np.maximum(count, 1)

    best = np.zeros((nt * nt, 2))
    best_s = np.full(nt * nt, -1.0)
    for (rng_, step) in (((-0.45, 0.45), 0.15), ((-0.1, 0.1), 0.05)):
        grid = np.arange(rng_[0], rng_[1] + 1e-9, step)
        start = best.copy()
        cand = best.copy()
        cand_s = np.full(nt * nt, -1.0)
        for dx in grid:
            for dy in grid:
                off = start + np.array([dx, dy])
                sc = score(off[tid, 0], off[tid, 1]) - 0.02 * np.hypot(off[:, 0], off[:, 1])   # on a tie, prefer a small shift
                better = sc > cand_s
                cand_s[better] = sc[better]
                cand[better] = off[better]
        best, best_s = cand, cand_s
    # tiles with too few cells: median of the neighbours; then light smoothing
    field = best.reshape(nt, nt, 2).astype(np.float32)
    valid = (count.reshape(nt, nt) >= 20).astype(np.float32)
    for ch in range(2):
        f = field[:, :, ch] * valid
        num = cv2.blur(f, (3, 3)) ; den = cv2.blur(valid, (3, 3))
        filled = np.where(valid > 0, field[:, :, ch], num / np.maximum(den, 1e-6))
        field[:, :, ch] = cv2.medianBlur(filled.astype(np.float32), 3)
    # shift per cell: bilinear over tile centres
    fx_ = (lat.xy[:, 0] + R) / tile - 0.5
    fy_ = (lat.xy[:, 1] + R) / tile - 0.5
    x0 = np.clip(np.floor(fx_).astype(int), 0, nt - 2)
    y0 = np.clip(np.floor(fy_).astype(int), 0, nt - 2)
    wx = np.clip(fx_ - x0, 0, 1)[:, None]
    wy = np.clip(fy_ - y0, 0, 1)[:, None]
    d = (field[y0, x0] * (1 - wx) * (1 - wy) + field[y0, x0 + 1] * wx * (1 - wy)
         + field[y0 + 1, x0] * (1 - wx) * wy + field[y0 + 1, x0 + 1] * wx * wy)
    return base + jx * d[:, 0:1] + jy * d[:, 1:2], field


def decode_all(img, parse: bool = True, fast: bool = True, _no_rescale: bool = False) -> list:
    """All codes in an image (e.g. a screen sheet with several circles) -> [(Chunk, DecodeInfo), ...].
    parse=False returns the raw payload bytes instead of the file chunk (for other containers, e.g. Strom)."""
    long_edge = max(img.shape[:2])
    if long_edge > 2200 and not _no_rescale:
        # Pre-pass on a downscaled image: find the bullseye, derive the working resolution from it
        # (more than ~4.5 camera pixels per cell gains nothing but costs time quadratically)
        f0 = 1600.0 / long_edge
        small0 = cv2.resize(img, None, fx=f0, fy=f0, interpolation=cv2.INTER_AREA)
        g0 = cv2.cvtColor(small0, cv2.COLOR_BGR2GRAY) if small0.ndim == 3 else small0
        c0 = [c for c in _find_rings(_binarize(g0)) if c["depth"] >= 2]
        if c0:
            ppm_full = max(c0, key=lambda c: c["depth"] * 1e9 + c["area"])["r_px"] / BULL_OUTER / f0
            f = min(1.0, 4.5 / ppm_full)
            if f < 1.0:
                img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
                res = decode_all(img, parse, fast, _no_rescale=True)
                for _, info in res:                      # convert image coordinates back to the original image
                    info.scale = f
                return res
        _no_rescale = True
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    binary = _binarize(gray)
    cands = _find_rings(binary)
    if not cands:
        raise ValueError("no bullseye found")
    maxd = max(c["depth"] for c in cands)
    if maxd < 2:
        raise ValueError("no bullseye found")
    bulls = sorted([c for c in cands if c["depth"] == maxd], key=lambda c: -c["area"])
    ppm0 = bulls[0]["r_px"] / BULL_OUTER
    if ppm0 > 5.5 and not _no_rescale:
        # more than ~5 camera pixels per cell gains nothing but costs time quadratically
        f = 4.5 / ppm0
        small = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        return decode_all(small, parse, fast, _no_rescale=True)
    # discard bullseyes that are only inner rings of a larger one
    keep = []
    for b in bulls:
        if all(np.linalg.norm(b["center"] - k["center"]) > 2 * k["r_px"] for k in keep):
            keep.append(b)
    results, errors = [], []
    for bull in keep:
        # candidates belong to the nearest bullseye (Voronoi)
        mine = [c for c in cands if c is not bull and
                all(np.linalg.norm(c["center"] - bull["center"]) <= np.linalg.norm(c["center"] - o["center"])
                    for o in keep if o is not bull)]
        try:
            raw, info = _decode_one(gray, bull, mine, fast, img if img.ndim == 3 else None)
            results.append((dicht.parse_chunk(raw) if parse else raw, info))
        except (ValueError, ReedSolomonError) as e:
            errors.append(str(e))
    if not results:
        raise ValueError(errors[0] if errors else "no readable code")
    return results


def decode_image(img, debug: bool = False):
    """Photo/scan (BGR or grey, uint8) -> (Chunk, DecodeInfo) of the first readable code."""
    return decode_all(img)[0]


def _decode_one(gray, bull, cands, fast: bool = True, bgr=None):
    ppm = bull["r_px"] / BULL_OUTER
    k = max(9, int(round(ppm * 10)) | 1)
    gray_n = _normalize_illum(gray, k)
    col_n = None                                    # colour image normalised per channel, only when needed
    center = _subpixel_center(gray_n, bull["center"], bull["r_px"])
    others = [c for c in cands if c is not bull and np.linalg.norm(c["center"] - center) > 14 * ppm]
    for c in others:
        c["center"] = _subpixel_center(gray_n, c["center"], c["r_px"])
        c["d"] = np.linalg.norm(c["center"] - center)
        c["ang"] = math.degrees(math.atan2(c["center"][1] - center[1], c["center"][0] - center[0]))
    # separate principal markers (large, 2 rings) and alignment markers (small, 1 ring) by size
    principal = [c for c in others if 0.72 < c["r_px"] / (PRINCIPAL_OUTER * ppm) < 1.4]
    rings = [c for c in others if 0.6 < c["r_px"] / (MARKER_OUTER * ppm) < 1.35]
    if len(principal) < 4:
        raise ValueError(f"only {len(principal)} principal markers found")
    principal.sort(key=lambda c: c["ang"])
    slot_of = {}
    for c in principal:
        sl = int(round(((c["ang"] - principal[0]["ang"]) % 360) / 60.0)) % 6
        if sl not in slot_of or slot_of[sl]["area"] < c["area"]:
            slot_of[sl] = c
    outer = [slot_of[sl] for sl in sorted(slot_of)]
    slots = sorted(slot_of)
    if len(outer) < 4:
        raise ValueError("principal markers cannot be assigned")
    rho_px = np.mean([c["d"] for c in outer])
    R_est = rho_px / ppm + MARKER_MARGIN
    info = DecodeInfo(ppm=ppm, center=center)
    imgf = gray_n.astype(np.float32)
    errors = []
    for R, shape in [(r, sh) for r in sorted(RADII, key=lambda r: abs(math.log(r / R_est)))[:3]
                     for sh in ("circle", "rect", "hexagon")]:
        lat = build_lattice(R, shape)
        mpos = lat.markers
        if not mpos:
            continue
        # ---- Geometry (rotation-independent, the marker layout has 6-fold symmetry):
        # homography from centre + principal markers with slot s -> principal marker j=s
        pmain = {m[1]: np.array(m[2]) for m in mpos if m[0] == 0}
        src = [[0.0, 0.0]]
        dst = [center]
        for c, s in zip(outer, slots):
            src.append(pmain[s])
            dst.append(c["center"])
        src, dst = np.array(src, np.float64), np.array(dst, np.float64)
        H, _ = cv2.findHomography(src.astype(np.float32), dst.astype(np.float32), 0)
        if H is None:
            continue
        Hinv = np.linalg.inv(H)
        # assign inner marker rings from the outside inwards; after each ring the
        # displacement field (TPS on the residual to the homography) is re-estimated so that
        # lens distortion and paper waves do not push the prediction out of the search radius
        ctrl_src, ctrl_dst = list(src), list(dst)
        pool = list(rings)
        used = set()
        tps = None

        def match_ring(ms, grid_pts):
            """Assign markers ms (grid coordinates grid_pts in the rot=0 frame), re-estimate the TPS.
            Returns the number of hits."""
            nonlocal tps
            gp_corr = grid_pts + _tps_eval(tps, grid_pts) if tps is not None else grid_pts
            pred = cv2.perspectiveTransform(gp_corr.reshape(-1, 1, 2), H).reshape(-1, 2)
            hits = 0
            for (g, p) in zip(grid_pts, pred):
                best = None
                for ci, c in enumerate(pool):
                    if ci in used:
                        continue
                    dd = np.linalg.norm(c["center"] - p)
                    if dd < 10.0 * ppm and (best is None or dd < best[0]):
                        best = (dd, ci)
                if best:
                    used.add(best[1])
                    ctrl_src.append(np.array(g))
                    ctrl_dst.append(pool[best[1]]["center"])
                    hits += 1
            cs, cd = np.array(ctrl_src), np.array(ctrl_dst)
            back = cv2.perspectiveTransform(cd.reshape(-1, 1, 2).astype(np.float64), Hinv).reshape(-1, 2)
            tps = _tps_fit(cs, back - cs, lam=1e-3)
            return hits

        ring_ks = sorted(set(m[0] for m in mpos if 0 < m[0] < 100))
        for ring_k in ring_ks:                       # rings with 6-fold symmetry: rotation-independent
            ms = [m for m in mpos if m[0] == ring_k]
            match_ring(ms, np.array([m[2] for m in ms], np.float64))
        rot_candidates = list(range(6))
        grid_ks = sorted(set(m[0] for m in mpos if m[0] >= 100))
        if grid_ks:                                  # rectangle: grid markers only have 2-fold symmetry,
            gm = [m for m in mpos if m[0] >= 100]    # so determine the rotation geometrically now
            gp_all = np.array([m[2] for m in gm], np.float64)
            pts_pool = np.array([c["center"] for c in pool]) if pool else np.zeros((0, 2))
            counts = []
            for rot in range(6):
                q = gp_all @ _rot(-60 * rot)[:2, :2].T
                q = q + _tps_eval(tps, q) if tps is not None else q
                pred = cv2.perspectiveTransform(q.reshape(-1, 1, 2), H).reshape(-1, 2)
                if len(pts_pool):
                    dmin = np.min(np.linalg.norm(pred[:, None, :] - pts_pool[None, :, :], axis=-1), axis=1)
                    counts.append(int((dmin < 10.0 * ppm).sum()))
                else:
                    counts.append(0)
            best_n = max(counts)
            rot_candidates = [r for r in range(6) if counts[r] == best_n] or list(range(6))
            rot_geo = rot_candidates[0]
            Rm = _rot(-60 * rot_geo)[:2, :2].T
            for gk in grid_ks:
                ms = [m for m in gm if m[0] == gk]
                match_ring(ms, np.array([m[2] for m in ms], np.float64) @ Rm)
        ctrl_dst = np.array(ctrl_dst)

        def to_image(pts_grid: np.ndarray, rot: int, tps_on: bool = True) -> np.ndarray:
            """Grid points (under rotation rot*60 degrees) -> image points, incl. displacement field."""
            q = pts_grid @ _rot(-60 * rot)[:2, :2].T
            if tps is not None and tps_on:
                q = q + _tps_eval(tps, q)
            return cv2.perspectiveTransform(q.reshape(-1, 1, 2).astype(np.float64), H).reshape(-1, 2)

        o = 0.18 * ppm
        offs = np.array([[0, 0], [o, 0], [-o, 0], [0, o], [0, -o]], np.float32)
        # reference levels from the bullseye (rotation-independent): outer black ring, white ring around it
        lat0 = build_lattice(R, "circle")
        blk = lat0.xy[(lat0.dist > 9.0) & (lat0.dist <= 10.25)]
        wht = lat0.xy[(lat0.dist > 10.25) & (lat0.dist <= 11.75)]
        rv = _sample(imgf, to_image(np.vstack([blk, wht]), 0), offs)
        thr = (np.median(rv[:len(blk)]) + np.median(rv[len(blk):])) / 2
        # ---- Rotations: read the format ring
        for rot in rot_candidates:
            if True:
                fv = _sample(imgf, to_image(lat.xy[lat.format_idx], rot), offs)
                reps = len(lat.format_idx) // 48
                tries = [fv[rep * 48:(rep + 1) * 48] for rep in range(reps)]
                if reps >= 2:
                    tries.append(np.mean(tries[:reps], axis=0))
                got = None
                for tv in tries:
                    try:
                        got = unpack_format((tv < thr).astype(np.uint8))
                        break
                    except (ReedSolomonError, ValueError):
                        continue
                if got is None or got[0] != shape or got[1] != R:
                    continue
                # ---- Hit: shape, radius, EC, rotation known -> read data
                _, _, ec_idx, levels, color = got
                info.R, info.shape, info.rotation, info.H = R, shape, rot, H @ _rot(-60 * rot)
                info.levels, info.color = levels, color
                if color:
                    if bgr is None:
                        errors.append("colour code, but grey image")
                        continue
                    if col_n is None:
                        col_n = np.stack([_normalize_illum(bgr[:, :, c], k) for c in range(3)], -1).astype(np.float32)
                info.markers = list(ctrl_dst)
                # Fast path: first without fine alignment (enough for screens and smooth paper),
                # only if Reed-Solomon then does not come through cleanly, repeat with fine alignment
                attempts = [False, True] if fast else [True]
                raw = None
                for refine in attempts:
                    ipts = _refine_grid(imgf, lat, to_image, rot)[0] if refine else to_image(lat.xy, rot)
                    v = _sample(imgf, ipts, offs)
                    if color:
                        # equalise the three channels (B, G, R in the OpenCV image) separately; value = R<<2|G<<1|B
                        chans = [_equalize(_sample(col_n[:, :, c], ipts, offs), lat, levels=2)[0] for c in range(3)]
                        cells = (chans[2].astype(np.uint8) << 2) | (chans[1].astype(np.uint8) << 1) | chans[0].astype(np.uint8)
                    else:
                        cells, coef = _equalize(v, lat, levels=levels)
                    fx = lat.fixed >= 0
                    fx[lat.format_idx] = False
                    fixed_err = float(((v[fx] < 0.5).astype(np.int8) != lat.fixed[fx]).mean())   # fixed cells are always b/w
                    if fixed_err > 0.15:
                        errors.append(f"R={R} {shape} rot={rot * 60}: fixed cells {fixed_err:.0%} wrong")
                        continue
                    try:
                        raw, errs = decode_cells(cells, R, shape, ec_idx, levels, color)
                        break
                    except (ReedSolomonError, ValueError) as e:
                        errors.append(f"R={R} {shape}{' (without fine alignment)' if not refine else ''}: {e}")
                if raw is None:
                    continue
                info.rs_errors = errs
                info.note = (f"R={R} {shape} {'colour' if color else f'{levels} levels'} ppm={ppm:.2f} rot={rot * 60} Marker={len(ctrl_dst)} "
                             f"fixed-cell-error={fixed_err:.1%} RS-corrections={errs}")
                return raw, info
    if errors:
        raise ValueError("data not decodable: " + " | ".join(errors))
    raise ValueError(f"format ring not readable (R estimated {R_est:.0f}, {len(outer)} outer markers)")


def draw_debug(img, info: DecodeInfo):
    out = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if info.center is not None:
        cv2.circle(out, tuple(np.round(info.center).astype(int)), 6, (0, 0, 255), 2)
    for m in info.markers:
        cv2.circle(out, tuple(np.round(m).astype(int)), 5, (0, 200, 0), 2)
    if info.H is not None and info.R:
        ang = np.linspace(0, 2 * np.pi, 120)
        if info.shape == "rect":
            W = info.R * RECT_ASPECT
            circ = np.array([[-W, -info.R], [W, -info.R], [W, info.R], [-W, info.R]], np.float32).reshape(-1, 1, 2)
        else:
            circ = np.stack([info.R * np.cos(ang), info.R * np.sin(ang)], 1).astype(np.float32).reshape(-1, 1, 2)
        ip = cv2.perspectiveTransform(circ, info.H).astype(np.int32)
        cv2.polylines(out, [ip], True, (255, 120, 0), 2)
    return out


# --------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="wabe. prototype")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capacity")
    c.add_argument("--ec", type=int, default=1)
    e = sub.add_parser("encode")
    e.add_argument("infile")
    e.add_argument("-o", "--out", default="wabe")
    e.add_argument("-r", "--radius", type=int, default=None)
    e.add_argument("--shape", default="circle", choices=list(SHAPES))
    e.add_argument("--style", default="hex", choices=["hex", "dot"])
    e.add_argument("--ppm", type=float, default=6)
    e.add_argument("--ec", type=int, default=1, help="0=15%% 1=25%% 2=35%% 3=50%%")
    e.add_argument("--levels", type=int, default=2, choices=[2, 4], help="2 = black/white, 4 = grey levels (2 bits per cell)")
    e.add_argument("--farbe", "--color", dest="farbe", action="store_true", help="3 channels per cell (screens only)")
    sc = sub.add_parser("screen", help="file -> screen sheets (PNG, display 1:1 and photograph)")
    sc.add_argument("infile")
    sc.add_argument("-o", "--out", default="blatt")
    sc.add_argument("--size", default="1920x1080")
    sc.add_argument("--px", type=float, default=3.5, help="screen pixels per cell (3.5 is enough, 4 is safer)")
    sc.add_argument("--levels", type=int, default=4, choices=[2, 4])
    sc.add_argument("--cols", type=int, default=2)
    st = sub.add_parser("strom", help="file -> frame sequence for the screen (fountain code), HTML player + GIF")
    st.add_argument("infile")
    st.add_argument("-o", "--out", default="strom")
    st.add_argument("-r", "--radius", type=int, default=128)
    st.add_argument("--px", type=float, default=None, help="screen pixels per cell (default 3.5, colour 4.5 because of JPEG chroma subsampling)")
    st.add_argument("--fps", type=float, default=4.0)
    st.add_argument("--frames", type=int, default=0, help="number of frames (0 = 4 x block count)")
    st.add_argument("--levels", type=int, default=2, choices=[2, 4])
    st.add_argument("--form", default="circle", choices=list(SHAPES), help="rect = screen 16:9, R = half the height")
    st.add_argument("--farbe", "--color", dest="farbe", action="store_true", help="3 channels per cell (screens only)")
    d = sub.add_parser("decode")
    d.add_argument("images", nargs="+")
    d.add_argument("-o", "--out", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "capacity":
        print(f"EC {EC_LEVELS[a.ec]:.0%}   (circle)")
        print(f"{'Radius':>7} {'Cells':>8} {'Bytes B/W':>10} {'4 levels':>9}  Diameter at 0.25 mm pitch")
        for R in RADII:
            n, _, K = capacity(R, "circle", EC_LEVELS[a.ec])
            K4 = capacity(R, "circle", EC_LEVELS[a.ec], 4)[2]
            print(f"{R:>7} {n:>8} {K:>10} {K4:>9}  {2 * R * 0.25:>5.0f} mm")
    elif a.cmd == "encode":
        data = open(a.infile, "rb").read()
        cells = encode_file(data, os.path.basename(a.infile), R=a.radius, shape=a.shape, ec_idx=a.ec, levels=a.levels,
                            color=a.farbe)
        R = next(r for r in RADII if len(build_lattice(r, a.shape).q) == len(cells[0]))
        for i, cl in enumerate(cells):
            fn = f"{a.out}.png" if len(cells) == 1 else f"{a.out}_{i + 1:03d}.png"
            im = render(cl, R, a.shape, a.ppm, style=a.style, color=a.farbe, levels=a.levels)
            cv2.imwrite(fn, cv2.cvtColor(im, cv2.COLOR_RGB2BGR) if a.farbe else im)
            print(fn, f"R={R}")
    elif a.cmd == "screen":
        w, h = (int(v) for v in a.size.lower().split("x"))
        data = open(a.infile, "rb").read()
        sheets, R, n = screen_sheets(data, os.path.basename(a.infile), w, h, a.px, a.levels, a.cols)
        for i, sh in enumerate(sheets):
            fn = f"{a.out}_{i + 1:02d}.png" if len(sheets) > 1 else f"{a.out}.png"
            cv2.imwrite(fn, sh)
            print(fn)
        print(f"{n} code(s) with R={R}, {a.levels} levels, on {len(sheets)} sheet(s)")
    elif a.cmd == "strom":
        import strom
        from PIL import Image
        data = open(a.infile, "rb").read()
        if a.px is None:
            a.px = 4.5 if a.farbe else 3.5
        if a.farbe and a.form == "rect" and a.radius == 128 and a.px >= 4.5:
            a.radius = 100                                   # fits on 1920x1080 at 4.5 px per cell
        enc = strom.StreamEncoder(data, os.path.basename(a.infile), R=a.radius, shape=a.form, levels=a.levels, color=a.farbe)
        n = a.frames or max(4, 4 * enc.K)
        cells = enc.frames(n)
        pngs = [strom.frame_png(c, a.radius, a.form, a.px, color=a.farbe) for c in cells]
        open(f"{a.out}.html", "w").write(strom.player_html(pngs, a.fps, os.path.basename(a.infile)))
        ims = [Image.open(io.BytesIO(p)).convert("RGB" if a.farbe else "L") for p in pngs]
        ims[0].save(f"{a.out}.gif", save_all=True, append_images=ims[1:], duration=int(1000 / a.fps), loop=0)
        print(f"{a.out}.html, {a.out}.gif: {n} frames, {enc.K} blocks of {enc.block_len} bytes, R={a.radius} {a.form}"
              f"{' colour' if a.farbe else ''}, "
              f"{a.fps} fps -> the file is available after {enc.K} frames read ({enc.K / a.fps:.1f} s)")
    elif a.cmd == "decode":
        col = dicht.Collector()
        for fn in a.images:
            for chunk, info in decode_all(cv2.imread(fn)):
                print(f"{fn}: Chunk {chunk.idx + 1}/{chunk.total} {info.note}")
                col.add(chunk)
        name, kind, data = col.result()
        out = a.out or (name or f"wabe_out.{kind}")
        open(out, "wb").write(data)
        print(f"-> {out} ({len(data)} bytes)")


if __name__ == "__main__":
    main()
