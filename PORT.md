# wabe. Decoder: Portierungs-Kit (Ziel: Rust oder C++ nach WebAssembly, läuft im Handy-Browser)

Ziel: 10 bis 20 Frames je Sekunde auf dem Handy ohne Server. Referenz ist `wabe.py` (Python/NumPy/OpenCV, ca. 1 s je Frame). `fixtures/` enthält Testbilder mit den erwarteten Nutzbytes (`erwartet.json`, SHA-256). Ein Port ist fertig, wenn alle sechs Bilder dieselben Nutzbytes liefern.

Reihenfolge, die sich bewährt hat: erst Gitter und Encoder (verlustfreier Roundtrip gegen `wabe.encode_cells`), dann den Decoder Stufe für Stufe gegen die Zwischenwerte der Referenz.

## 1. Gitter

* Sechseckgitter, Zellmitte in axialen Koordinaten (q, r): `x = q + r/2`, `y = r·√3/2`. Nachbarabstand 1 (= 1 Pitch). Nachbarn: (±1,0), (0,±1), (1,−1), (−1,1).
* Punkt → Zelle: `q = x − y/√3`, `r = 2y/√3`, dann cube rounding (siehe `xy_to_axial_round`).
* Umriss: Kreis `hypot(x,y) ≤ R`; Sechseck `max(|q|,|r|,|q+r|) ≤ R`; Rechteck `|x| ≤ R·16/9, |y| ≤ R`. R ∈ RADII = [32, 40, 50, 64, 80, 100, 128, 160, 200, 256, 320, 400, 500].
* Zellreihenfolge für Daten: alle nicht festen Zellen, sortiert nach (round(Abstand zur Mitte), Winkel atan2(y,x)); stabile Sortierung.
* Feste Zellen (Ringe sind (vorheriger Radius, Radius]): Bullseye [(2,s),(3.5,w),(4.75,s),(6.25,w),(7.5,s),(9,w),(10.25,s),(11.75,w)]; Formatring (11.75, 13.75]; Hauptmarker [(1.5,s),(2.75,w),(4,s),(5.25,w),(6.5,s),(7.75,w)]; Ausrichtungsmarker [(1.5,s),(2.75,w),(4,s),(5,w)]. Die erste Schale schließt den Abstand 0 ein.
* Markerpositionen: `marker_positions(R, form)`. Hauptmarker j=0..5 bei Winkel 60j, Radius ρ0 = R−8 (Sechseck: R·√3/2 − 8). Ausrichtungsringe bei ρ0, ρ0−40, … (≥ 24), n = 6·max(1, floor(2πρ/240 + 0.5)) Marker, Ring 1 lässt die Hauptmarkerplätze aus, gerade Ringe sind um 180/n Grad versetzt. Rechteck zusätzlich: Gitter alle 40 Pitch außerhalb ρ0+12 mit Mindestabstand 14 zu anderen Markern, Ring-Index 100+Band.
* Kontrollpunkt eines Markers = Tintenschwerpunkt seiner schwarzen Zellen innerhalb des vorletzten Radius (nicht die Sollposition).

## 2. Bits und Codewörter

* Bits je Zelle: 1 (s/w), 2 (4 Stufen: Wert = 2·b0 + b1, 0 weiß … 3 schwarz), 3 (Farbe: Wert = R<<2 | G<<1 | B mit 1 = Tinte im Kanal; Bitfolge R, G, B).
* nbits = Datenzellen × Bits je Zelle, B = nbits // 8 Codewort-Bytes. Füllbits nach den Daten: Zufall (egal beim Lesen).
* RS-Plan: nb = ceil(B/255) Blöcke, Längen base/base+1 (die ersten B mod nb Blöcke länger), nsym = clamp(round(n·ec), 2, n−1), ec ∈ {0.15, 0.25, 0.35, 0.5}. Reed-Solomon GF(256), Primpolynom 0x11D, Generator 2, fcr 0, msg[0] = höchster Koeffizient (reedsolo-Konvention; Syndrome j: Σ c_i·α^(j·deg_i)).
* Verschachtelung: Byte t des Zellstroms = Block (t mod nb)… genau: für p in 0..maxn−1, für jeden Block b mit p < n_b: nächstes Byte = block_b[p] (`interleave_order`).
* Scrambler: XOR mit xorshift32, Seed 0x6D6B8A3F, je Byte: x ^= x<<13; x ^= x>>17; x ^= x<<5; Byte = x & 0xFF (alles mod 2^32).
* Formatring: 16 Bit `v = ver(2) | form(3)<<2 | radiusindex(5)<<5 | ec(2)<<10 | stufen4(1)<<12 | farbe(1)<<13`, dann CRC-8 (Poly 0x07, Init 0) über die 2 Bytes, dann RS mit nsym 3 → 6 Bytes = 48 Bit, MSB zuerst, in den Formatring-Zellen (sortiert nach Winkel) dreimal hintereinander. Formatzellen sind schwarz für Bit 1.
* Datei-Container (`dicht.py`): Header `<2sBBHHIIIB` = Magic "dc", Version 1, flags (Typ | Kompression<<4; 1 = Brotli, 2 = LZMA), Chunk-Index, Chunk-Anzahl, Gesamtlänge komprimiert, CRC32 der Datei, Payload-Länge, Namenslänge, dann Name, dann Payload.
* Strom (`strom.py`): Header `<2sBBIHHIIIB` = "ws", Version 1, flags, stream_id, K, block_len, total_len, crc32, seed, name_len, Name, dann block_len Bytes. Blockauswahl aus seed: NumPy `default_rng(seed).integers(0, 2, K)` wiederholt bis nicht leer. Für den Port den Generator ersetzen (z.B. xorshift) und Encoder und Decoder gemeinsam umstellen.

## 3. Decoder-Pipeline (Referenz `decode_all`, `_decode_one`)

1. Vorlauf: Bild auf ≤ 1600 px, Bullseye suchen, Arbeitsauflösung so, dass 4.5 px je Zelle (`ppm`).
2. Binärbild: `bg = blur(dilate(gray, k), k)` mit k = max(15, min(h,w)/25 ungerade); schwarz = gray/bg < 0.55.
3. Konturbaum (RETR_TREE). Kandidat = Außenrand einer schwarzen Fläche (gerade Verschachtelungsebene) mit Loch und schwarzem Kern darin (Fläche > 2 % der Außenfläche), Kern zentriert (< 0.2·√Fläche), konvexe Hülle rund (4πA/U² ≥ 0.6, Fläche ≥ 0.6·Hüllenfläche). `depth` = Anzahl verschachtelter Kerne. Bullseye = größte Tiefe, ppm = r_px/10.25 mit r_px = √(A/π). Hauptmarker: 0.72 < r_px/(6.5·ppm) < 1.4, Ausrichtungsmarker: 0.6 < r_px/(4·ppm) < 1.35, beide weiter als 14·ppm von der Mitte. Mitten per Tintenschwerpunkt im Umkreis 1.05·r_px+1 auf dem normierten Graubild.
4. Beleuchtungsnormierung: zwei Iterationen „Mittel der Werte unter/über der lokalen Schwelle“ mit Box-Filter k = 10·ppm (auf 1/4 gerechnet), `v = (g − low)/max(high − low, 12)`.
5. Hauptmarker nach Winkel in 60-Grad-Slots; Homographie (DLT, kleinste Quadrate) aus Mitte + Hauptmarkern mit Slot s → Hauptmarker j = s. Radius aus mittlerem Markerabstand/ppm + 8, die drei nächsten RADII probieren, je Form.
6. Ausrichtungsmarker ringweise von innen nach außen zuordnen (nächster Kandidat innerhalb 10·ppm der Vorhersage), nach jedem Ring Thin-Plate-Spline (Kernel r²·ln r, λ = 1e-3) auf dem Residuum in Gitterkoordinaten. Rechteck: Drehlage über Trefferzahl der Gittermarker bei den 6 Drehungen.
7. Abbildung Gitter → Bild: `q = Rot(−60·rot)·p; q += TPS(q); H·q`. TPS auf Gitter mit Schritt 4 Pitch auswerten, dazwischen bilinear.
8. Formatring: Schwellwert = Mitte zwischen Median des äußeren schwarzen Bullseye-Rings (9, 10.25] und des weißen Rings (10.25, 11.75]; drei Kopien und deren Mittel probieren; Form und R müssen zum Versuch passen.
9. Abtastung je Zelle: Mittel aus 5 Punkten (Mitte, ±0.18·ppm in x und y) bilinear. Farbe: je Kanal eigenes normiertes Bild.
10. Entzerrung: `v ≈ a·s + b·Σ(6 Nachbarn s) + c` per kleinster Quadrate (s = Tinte 0/1, fehlende Nachbarn = Papier), `s' = ((v − b·Σs − c)/a > 0.5)`, 2 Iterationen (4 Stufen: s ∈ {0, ⅓, ⅔, 1}, vorher v = (v+0.25)/1.5).
11. Plausibilität: feste Zellen (ohne Formatring) mit v < 0.5 müssen zu ≥ 85 % stimmen.
12. Schnellpfad: erst ohne Feinabgleich; schlägt RS fehl, Feinabgleich (`_refine_grid`: je 16×16-Pitch-Kachel Verschiebung mit maximalem Mittel |v − 0.5|, Suche ±0.45 Schritt 0.15, dann ±0.1 Schritt 0.05, kleine Verschiebung bevorzugt, Median 3×3, bilinear je Zelle).
13. Bits → Descramble → De-Interleave → RS (Syndrome vektorisiert, nur fehlerhafte Blöcke voll) → Container oder Strom.

## 4. Was im Port anders sein darf

* Konturen: ein eigener Connected-Components-Baum reicht (Außenrand, Loch, Kern).
* Homographie und TPS: kleine dichte lineare Systeme, Gauß mit Pivot reicht.
* Reed-Solomon: jede GF(256)-Implementierung mit demselben Primpolynom; Syndrom-Vorprüfung lohnt sich.
* Festkomma statt Float ist nicht nötig; f32 reicht überall.
* Erwartete Laufzeit nativ: Kreis R=128 unter 30 ms, Farb-Rechteck R=128 unter 100 ms.

## 5. Testen

```
python wabe.py decode fixtures/kreis_sw_papier.jpg -o /tmp/x.pdf      # Referenz
python - <<'EOF'
import json, hashlib, cv2, wabe
for c in json.load(open("fixtures/erwartet.json")):
    raw, info = wabe.decode_all(cv2.imread("fixtures/" + c["datei"]), parse=False)[0]
    print(c["datei"], hashlib.sha256(raw).hexdigest() == c["nutzbytes_sha256"], info.note)
EOF
```

Zwischenwerte zum Vergleichen stehen in `erwartet.json` (R, Form, Farbe, Stufen, Drehung, ppm, Markerzahl, Bullseye-Mitte in Pixeln des Originalbilds, Arbeitsmaßstab des Vorlaufs).
