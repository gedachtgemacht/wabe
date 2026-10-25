# wabe. decoder: porting kit (target: Rust or C++ to WebAssembly, runs in the phone browser)

Goal: 10 to 20 frames per second on the phone without a server. The reference is `wabe.py` (Python/NumPy/OpenCV, about 1 s per frame). `fixtures/` contains test images with the expected payload bytes (`erwartet.json`, SHA-256). A port is done when all six images yield the same payload bytes.

Order that has worked: first grid and encoder (lossless roundtrip against `wabe.encode_cells`), then the decoder stage by stage against the reference's intermediate values.

**Warning: lattice tie-breaking.** `build_lattice` decides cells whose distance to a marker centre equals a ring radius exactly by floating-point noise from `sin`/`cos`. glibc and musl (WebAssembly) disagree on a few cells; one cell of difference shifts the whole bit sequence. The Python decoder in the browser therefore loads the natively built grids from `web/py/lat.pkl` (see `tools/export_lat.py`). A native port has to reproduce exactly these tie decisions, e.g. by embedding the `fixed` tables for the supported radii/shapes, or by an explicit, versioned tie rule used by encoder and decoder alike.

## 1. Grid

* Hexagonal grid, cell centre in axial coordinates (q, r): `x = q + r/2`, `y = r·√3/2`. Neighbour distance 1 (= 1 pitch). Neighbours: (±1,0), (0,±1), (1,−1), (−1,1).
* Point → cell: `q = x − y/√3`, `r = 2y/√3`, then cube rounding (see `xy_to_axial_round`).
* Outline: circle `hypot(x,y) ≤ R`; hexagon `max(|q|,|r|,|q+r|) ≤ R`; rectangle `|x| ≤ R·16/9, |y| ≤ R`. R ∈ RADII = [32, 40, 50, 64, 80, 100, 128, 160, 200, 256, 320, 400, 500].
* Cell order for data: all non-fixed cells, sorted by (round(distance to centre), angle atan2(y,x)); stable sort.
* Fixed cells (rings are (previous radius, radius]): bullseye [(2,b),(3.5,w),(4.75,b),(6.25,w),(7.5,b),(9,w),(10.25,b),(11.75,w)]; format ring (11.75, 13.75]; principal markers [(1.5,b),(2.75,w),(4,b),(5.25,w),(6.5,b),(7.75,w)]; alignment markers [(1.5,b),(2.75,w),(4,b),(5,w)]. The first shell includes distance 0.
* Marker positions: `marker_positions(R, shape)`. Principal marker j=0..5 at angle 60j, radius ρ0 = R−8 (hexagon: R·√3/2 − 8). Alignment rings at ρ0, ρ0−40, … (≥ 24), n = 6·max(1, floor(2πρ/240 + 0.5)) markers, ring 1 skips the principal marker positions, even rings are offset by 180/n degrees. Rectangle additionally: a grid every 40 pitches outside ρ0+12 with a minimum distance of 14 to other markers, ring index 100+band.
* Control point of a marker = ink centroid of its black cells within the second-to-last radius (not the nominal position).

## 2. Bits and codewords

* Bits per cell: 1 (b/w), 2 (4 levels: value = 2·b0 + b1, 0 white … 3 black), 3 (colour: value = R<<2 | G<<1 | B with 1 = ink in the channel; bit order R, G, B).
* nbits = data cells × bits per cell, B = nbits // 8 codeword bytes. Padding bits after the data: random (irrelevant when reading).
* RS plan: nb = ceil(B/255) blocks, lengths base/base+1 (the first B mod nb blocks longer), nsym = clamp(round(n·ec), 2, n−1), ec ∈ {0.15, 0.25, 0.35, 0.5}. Reed-Solomon GF(256), primitive polynomial 0x11D, generator 2, fcr 0, msg[0] = highest coefficient (reedsolo convention; syndrome j: Σ c_i·α^(j·deg_i)).
* Interleaving: byte t of the cell stream = block (t mod nb)… precisely: for p in 0..maxn−1, for each block b with p < n_b: next byte = block_b[p] (`interleave_order`).
* Scrambler: XOR with xorshift32, seed 0x6D6B8A3F, per byte: x ^= x<<13; x ^= x>>17; x ^= x<<5; byte = x & 0xFF (everything mod 2^32).
* Format ring: 16 bit `v = ver(2) | form(3)<<2 | radiusindex(5)<<5 | ec(2)<<10 | levels4(1)<<12 | colour(1)<<13`, then CRC-8 (poly 0x07, init 0) over the 2 bytes, then RS with nsym 3 → 6 bytes = 48 bit, MSB first, into the format-ring cells (sorted by angle) three times in a row. Format cells are black for bit 1.
* File container (`dicht.py`): header `<2sBBHHIIIB` = magic "dc", version 1, flags (type | compression<<4; 1 = Brotli, 2 = LZMA), chunk index, chunk count, total compressed length, CRC32 of the file, payload length, name length, then name, then payload.
* Stream (`strom.py`): header `<2sBBIHHIIIB` = "ws", version 1, flags, stream_id, K, block_len, total_len, crc32, seed, name_len, name, then block_len bytes. Block selection from the seed: NumPy `default_rng(seed).integers(0, 2, K)`, repeated until non-empty. For the port, replace the generator (e.g. xorshift) and switch encoder and decoder together.

## 3. Decoder pipeline (reference `decode_all`, `_decode_one`)

1. Preliminary pass: image to ≤ 1600 px, find the bullseye, working resolution such that there are 4.5 px per cell (`ppm`).
2. Binary image: `bg = blur(dilate(gray, k), k)` with k = max(15, min(h,w)/25 odd); black = gray/bg < 0.55.
3. Contour tree (RETR_TREE). Candidate = outer edge of a black area (even nesting level) with a hole and a black core inside (area > 2 % of the outer area), core centred (< 0.2·√area), convex hull round (4πA/U² ≥ 0.6, area ≥ 0.6·hull area). `depth` = number of nested cores. Bullseye = greatest depth, ppm = r_px/10.25 with r_px = √(A/π). Principal marker: 0.72 < r_px/(6.5·ppm) < 1.4, alignment marker: 0.6 < r_px/(4·ppm) < 1.35, both further than 14·ppm from the centre. Centres by ink centroid in the circumscribed circle 1.05·r_px+1 on the normalised grey image.
4. Illumination normalisation: two iterations of "mean of values below/above the local threshold" with box filter k = 10·ppm (computed at 1/4 size), `v = (g − low)/max(high − low, 12)`.
5. Sort principal markers by angle into 60-degree slots; homography (DLT, least squares) from the centre + principal markers with slot s → principal marker j = s. Radius from the mean marker distance/ppm + 8, try the three nearest RADII, per shape.
6. Match alignment markers ring by ring from the inside out (nearest candidate within 10·ppm of the prediction), after each ring a thin-plate spline (kernel r²·ln r, λ = 1e-3) on the residual in grid coordinates. Rectangle: rotation from the hit count of the grid markers over the 6 rotations.
7. Grid → image mapping: `q = Rot(−60·rot)·p; q += TPS(q); H·q`. Evaluate the TPS on a grid with step 4 pitch, bilinear in between.
8. Format ring: threshold = midpoint between the median of the outer black bullseye ring (9, 10.25] and the white ring (10.25, 11.75]; try three copies and their mean; shape and R must fit the attempt.
9. Sampling per cell: mean of 5 points (centre, ±0.18·ppm in x and y), bilinear. Colour: own normalised image per channel.
10. Equalisation: `v ≈ a·s + b·Σ(6 neighbours s) + c` by least squares (s = ink 0/1, missing neighbours = paper), `s' = ((v − b·Σs − c)/a > 0.5)`, 2 iterations (4 levels: s ∈ {0, ⅓, ⅔, 1}, beforehand v = (v+0.25)/1.5).
11. Plausibility: fixed cells (without the format ring) with v < 0.5 must match at ≥ 85 %.
12. Fast path: first without fine alignment; if Reed-Solomon then fails, with fine alignment (`_refine_grid`: per 16×16-pitch tile the shift with maximal mean |v − 0.5|, search ±0.45 step 0.15, then ±0.1 step 0.05, small shift preferred, median 3×3, bilinear per cell).
13. Bits → descramble → de-interleave → RS (syndromes vectorised, full decoding only for faulty blocks) → container or stream.

## 4. What may differ in the port

* Contours: your own connected-components tree is enough (outer edge, hole, core).
* Homography and TPS: small dense linear systems, Gauss with pivoting is enough.
* Reed-Solomon: any GF(256) implementation with the same primitive polynomial; a syndrome pre-check pays off.
* Fixed point instead of float is not needed; f32 is enough everywhere.
* Expected native run time: circle R=128 under 30 ms, colour rectangle R=128 under 100 ms.

## 5. Testing

```
python wabe.py decode fixtures/kreis_sw_papier.jpg -o /tmp/x.pdf      # reference
python - <<'PY'
import json, hashlib, cv2, wabe
for c in json.load(open("fixtures/erwartet.json")):
    raw, info = wabe.decode_all(cv2.imread("fixtures/" + c["datei"]), parse=False)[0]
    print(c["datei"], hashlib.sha256(raw).hexdigest() == c["nutzbytes_sha256"], info.note)
PY
```

Intermediate values to compare are in `erwartet.json` (R, shape, colour, levels, rotation, ppm, marker count, bullseye centre in pixels of the original image, working scale of the preliminary pass).
