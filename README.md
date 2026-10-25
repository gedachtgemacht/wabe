# wabe. (Arbeitsname) – Prototyp v0.1

Runder Schwarz/Weiß-Code auf Sechseckgitter. Ganze Dateien (PDF, HTML, ...) im Code, ohne Server. Druckbar auf jedem Laser-, Tintenstrahl- oder Etikettendrucker, lesbar mit Handykamera oder Scanner.

## Dateien

| Datei | Zweck |
|---|---|
| `wabe.py` | Encoder, Decoder, CLI |
| `dicht.py` | Datei-Container, Reed-Solomon-Plan, Scrambler, Interleaving (geteilt) |
| `kamera.py` | Simulation Handyfoto und Scanner (Perspektive, Linse, Papierwelle, Unschärfe, Licht, Rauschen, JPEG) |
| `demo.py` | Erzeugt die Demo-Dateien in `demo/` |

`pip install numpy opencv-python reedsolo brotli reportlab pillow`

```
python wabe.py capacity
python wabe.py encode datei.pdf -o code            # -> code.png
python wabe.py encode datei.pdf --shape hexagon    # Sechseck
python wabe.py encode datei.pdf --levels 4         # 4 Graustufen, doppelte Kapazität (siehe unten)
python wabe.py decode foto.jpg
```

## Aufbau

* Zellen auf einem Sechseckgitter, Abstand der Zellmitten = 1 Pitch. 15 % mehr Zellen als ein Quadratgitter bei gleichem Mindestabstand (also gleicher Kameraauflösung).
* Umriss Kreis oder Sechseck, Radius R in Pitch. Jeder andere konvexe Umriss wäre mit derselben Maschinerie möglich.
* Bullseye in der Mitte: 4 schwarze Ringe, rotationsinvariant, in jeder Lage zu finden.
* 6 Hauptmarker (2 Ringe) am Rand: Perspektive. Orientierung kommt aus dem Formatring, der Decoder probiert die 6 Drehlagen durch.
* Ausrichtungsmarker (1 Ring) auf Ringen alle 40 Pitch, dicht genug für Linsenverzerrung und gewelltes Papier. Marker sitzen auf dem Tintenschwerpunkt ihrer Zellen, nicht auf der Sollposition (Gitterquantisierung bis 0,5 Pitch).
* Formatring um das Bullseye: Version, Form, Radius, EC-Stufe, Stufenzahl. 16 Bit + CRC-8 + RS(6,3), dreimal.
* Daten: Reed-Solomon GF(256), Redundanz 15/25/35/50 % wählbar, Blöcke verschachtelt, xorshift-Scrambler, Brotli-Kompression, Header mit Dateityp, Name, Chunk i/n, CRC32.
* Optional 4 Graustufen je Zelle = 2 Bit (für Bildschirm und Laserdruck; auf Thermodruckern bleibt es bei 2 Stufen).

## Decoder

1. Beleuchtungsnormiertes Binärbild, Konturbaum: Ringscheiben nach Verschachtelungstiefe und Größe.
2. Homographie aus Mitte und Hauptmarkern, Radius aus dem Markerabstand geschätzt.
3. Ausrichtungsmarker ringweise von außen nach innen zuordnen, nach jedem Ring Thin-Plate-Spline auf dem Residuum.
4. Formatring lesen (6 Drehlagen, Schwellwert aus den Bullseye-Ringen).
5. Datengetriebener Feinabgleich: je 16x16-Pitch-Kachel die Sub-Pitch-Verschiebung mit der bimodalsten Abtastung.
6. Lokale Schwarz/Weiß-Pegel aus den Daten selbst, Entzerrung des Übersprechens der 6 Nachbarn, Quantisierung.
7. Plausibilität über die festen Zellen, dann Descramble, De-Interleave, Reed-Solomon, Header, CRC.

## Kapazität (25 % Redundanz, Kreis)

| Radius | Zellen | Durchmesser (0,25 mm) | Bytes S/W | Bytes 4 Stufen |
|---|---|---|---|---|
| 32 | 3.697 | 16 mm | 164 | 328 |
| 64 | 14.845 | 32 mm | 1.205 | 2.417 |
| 100 | 36.295 | 50 mm | 3.124 | 6.244 |
| 160 | 92.845 | 80 mm | 8.081 | 16.140 |
| 256 | 237.727 | 128 mm | 21.090 | 42.107 |
| 320 | 371.485 | 160 mm | 32.814 | 65.629 |
| 400 | 580.381 | 200 mm | 51.365 | 102.794 |
| 500 | 906.901 | 250 mm | 80.335 | 160.670 |

QR Version 40: 2.953 Bytes bei 177x177 Modulen. Ehrlich gerechnet ist die Dichte je Zelle bei gleicher Redundanz ähnlich wie bei QR; der Gewinn kommt aus 15 % Sechseckgitter, aus der Größe (QR endet bei 177 Modulen), aus 2,6 bis 3 Kamerapixeln je Pitch statt 3 bis 4, aus Kompression, und aus dem 4-Stufen-Modus.

## Gemessen (Simulation)

* Auflösungsgrenze: 2,6 px/Pitch zuverlässig, 2,3 teilweise, darunter nichts. 4 Stufen: ab 4 px/Pitch zuverlässig.
* A4-Code (R=320, 32,8 KB) mit 12-MP-Handyfoto (3,6 px/Pitch): 20/20 bei zufälliger Perspektive bis 14 %, Drehung beliebig, Unschärfe bis 1,1 px, JPEG 55 bis 90, Linse k1 −0,1 bis 0,12, Papierwelle bis 12 px. Median 15 korrigierte Bytes von 44.000.
* Kleiner Code (R=100, 3,1 KB) unter denselben Bedingungen mit realistischer Papierwölbung: 38/40. Die zwei Ausfälle bei extremer Kissenverzerrung (k1 = −0,11).
* Linse k1 = 0,15 plus Welle 15 px auf A4: lesbar mit 999 Korrekturen (Grenze liegt bei 5.500).

## Bildschirm

Der A4-Code ist für Papier. Auf einem Laptop zeigt der PDF-Viewer die 640 Zellen Durchmesser auf etwa 1000 Bildschirmpixel, also 1,6 px je Zelle. Da sind die Zellen schon weg, bevor die Kamera dran ist (Grenze 2,6). Für den Bildschirm:

```
python wabe.py screen datei.pdf --size 1920x1080      # -> blatt_01.png, blatt_02.png ...
```

Erzeugt Blätter mit je bis zu 2 Kreisen (R=100, 4 Stufen, 3,5 px je Zelle). Blatt 1:1 anzeigen (Bildbetrachter auf 100 %, kein Browser-Zoom), ein Foto je Blatt, `decode` liest alle Kreise eines Fotos. Simulation mit Pixelraster, Moiré und Lichtreflex: 15/15 Codes bei 1,5 bis 2,5 Kamerapixeln je Bildschirmpixel.

## Strom: Datei vom Bildschirm aufs Handy durch Draufhalten

Die kostenlose Dimension am Bildschirm ist die Zeit. `strom` zerlegt die Datei in K Blöcke und zeigt eine Schleife von Frames; jeder Frame ist eine zufällige XOR-Kombination der Blöcke (Random Linear Fountain über GF(2), Seed im Frame). Das Handy muss weder den ersten Frame erwischen noch jeden: irgendwelche K bis K+2 lesbare Frames reichen, egal in welcher Reihenfolge. Kein Sync, kein Rückkanal.

```
python wabe.py strom datei.pdf -o strom          # -> strom.html (Player, Geraetepixel 1:1) und strom.gif
```

Standard: R=128, 5,1 KB je Frame, 4 fps. Das 42-KB-Handbuch sind 7 Blöcke, also 1,8 s Draufhalten bei verlustfreiem Lesen; in der Simulation mit abgefilmtem Bildschirm, zufälligem Start und jedem dritten Frame verpasst waren es 7 bis 8 Fotos. Decoder braucht 0,4 s je Frame (1500 px Foto) in Python, also 2 bis 3 Frames je Sekunde, 10 bis 15 KB/s. `server_beispiel.py` zeigt die FastAPI-Anbindung: Handy schickt alle 300 bis 500 ms ein verkleinertes JPEG, Server meldet den Fortschritt, am Ende kommt die Datei.

Die Mathematik dahinter in `strom.py` (StreamEncoder, StreamDecoder mit inkrementeller Gauss-Elimination).

## Farbe und Rechteck (Bildschirm)

`--farbe`: drei Kanäle je Zelle, Wert = R<<2 | G<<1 | B mit 1 = Tinte im Kanal (0 weiß, 7 schwarz). Alle festen Strukturen bleiben schwarz/weiß, die Erkennung läuft unverändert auf dem Graubild; nur die Datenzellen werden je Kanal getrennt normiert, entzerrt und entschieden. `--form rect`: 16:9-Rechteck, R = halbe Höhe, Ausrichtungsmarker auf einem 40-Pitch-Gitter außerhalb des Innenkreises; die Drehlage bestimmt der Decoder dort geometrisch (Gittermarker sind nur 2-zählig symmetrisch), der Formatring entscheidet den Rest.

Ein Bildschirm-Rechteck R=128 (1614×917 px bei 3,5 px je Zelle) in Farbe fasst **35 KB je Frame**, 7x gegenüber dem S/W-Kreis und 6x gegenüber zwei QR v40 nebeneinander. Simulation (Pixelraster, Moiré, Reflex, Farbstich, Perspektive, beliebige Drehung): 12/12, median 21 korrigierte Bytes. Der 160-KB-Katalog war nach 4 abgefilmten Frames komplett. Decode 1,2 bis 1,5 s je Frame in Python (134.000 Zellen, 3 Kanäle).

```
python wabe.py strom datei.pdf --form rect --farbe -o strom     # Vollbild, Farbe
python wabe.py encode datei.pdf --shape rect --farbe            # ein einzelnes Bildschirmbild (bis 35 KB)
```

Praxis: Das Handy exportiert JPEG mit Chroma-Subsampling 4:2:0 (halbe Farbauflösung), und genau in der Farbe stecken die Bits. Deshalb Farbzellen mindestens 4,5 Bildschirmpixel groß machen (`strom --farbe` nimmt das als Standard, R=100, 21,7 KB je Frame): in der Simulation 0 Korrekturen statt bis zu 734 bei 3,5 px. Farbe ist nur für Bildschirme gedacht; auf Papier gelten Druckerfarben und Chroma-Subsampling, dort bleibt S/W.

## Vergleich mit QR im Live-Modus

Gleiche Bildschirmfläche (917 px), gleicher Fountain-Code, gleiche Bildschirm-Simulation, Handbuch 42 KB:

| | Bytes je Frame | Frames bis fertig | Decode je Foto | Lesbar (Sim) |
|---|---|---|---|---|
| QR Version 40, EC L (7 %), zxing-cpp | 2.916 | 11 bis 12 | 10 bis 13 ms (C++) | 6/6 |
| wabe R=128, EC 25 %, Python | 5.093 | 7 bis 8 | ca. 250 ms (Python) | 6/6 |

Pro Frame trägt wabe 1,7x (bei gleicher Redundanz 3x). Live zählt aber der ganze Zyklus Aufnahme, JPEG, Upload, Decode: da liegt QR heute gleichauf bis vorne, und mit Decoder im Browser (zxing-wasm, kein Upload) deutlich vorne. Der Grund ist der 25 Jahre alte, in C++ optimierte Decoder, nicht der Code. `qrstrom.py` liefert dieselbe Strom-Logik mit QR-Frames für den A/B-Test auf dem eigenen Handy: `qrstrom.qr_stream()` erzeugt Frames, `qrstrom.decode_qr_payloads(img)` liest sie, die Payloads gehen in denselben `StreamDecoder`.

## Live-Seite (server.py + web/index.html)

Der Button **Video** schickt laufend Kamerabilder an `POST /api/strom` (volle Kamerabreite, Hoehe mittig auf 80 % der Breite, 3000 px, JPEG; drei Anfragen gleichzeitig). Der Server (4 Decode-Slots) loest die Frames mit `strom.StreamDecoder` und meldet Bloecke, Fortschritt und `ppm` (Kamerapixel je Zelle); die fertige Datei kommt ueber `/api/datei`. Diagnose: `GET /api/strom/log` (Zeile je Frame mit Kameragroesse, Schaerfe, ppm, Fehler) und `GET /api/strom/bilder[?nr=N]` (die letzten 15 unlesbaren Frames, nur RAM). Gemessen: unter ca. 4,5 Kamerapixeln je Zelle scheitert das Lesen; das Handy-JPEG (Chrome, 4:2:0) kostet bei Farbe die meisten Reserven.

### App und Offline (Lokal-Modus)

`web/` ist eine PWA: `manifest.webmanifest` + Icons (installierbar), `sw.js` (Service Worker haelt Seite, `lokal.js`, `py/` und das selbst gehostete Pyodide in `pyodide/` im Cache, ca. 28 MB). Der Button **Lokal** decodiert im Browser (2 Pyodide-Worker, `?w=3` aendert die Zahl; auf dem Handy war 3 langsamer als 2), **Video** schickt Bilder an den Server. `py/lat.pkl` enthaelt die nativ gebauten Gitter (Rand-Rundung weicht in WebAssembly ab), bei neuen Radien/Formen neu exportieren. Deploy: `./deploy.sh` (setzt die Service-Worker-Version aus einem Hash der Quellen). Einrichten nach dem Klonen: `tools/pyodide_holen.sh` laedt Pyodide nach `web/pyodide/` (nicht im Repo), `tools/export_lat.py` baut `web/py/lat.pkl` neu.

## Offen

* Echter Druck- und Fototest. Vor allem: Tonwertzuwachs im 4-Stufen-Modus (Mittelstufen brauchen Kalibrierzellen).
* Kalibrierzellen für den 4-Stufen-Modus (Display-Gamma und Kamera-Tonkurve verschieben die Mittelstufen, deshalb ist dein Bildschirmfoto gescheitert; S/W liest).
* Erasure-Decoding für unsichere Zellen.
* Decoder nativ (Rust/C++ nach WASM): dann kein Upload, 10 bis 20 Frames je Sekunde, 50 bis 100 KB/s. Das ist die Bedingung, unter der wabe im Live-Modus an QR vorbeizieht.
* Decode-Zeit: 6 bis 17 s für A4 in Python, in C/WASM unter einer Sekunde.

## Lizenz

MIT, siehe `LICENSE`. Drittcode: `THIRD_PARTY.md`.
