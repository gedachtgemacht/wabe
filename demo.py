"""Demo files for wabe.: codes (circle, hexagon, 4 levels), simulated photos with detection, A4 page."""
import os
import subprocess
import time

import numpy as np
import cv2
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

import wabe
import kamera
import dicht

OUT = "demo"
os.makedirs(OUT, exist_ok=True)
SRC = "../dicht/demo"
ls = open(f"{SRC}/lieferschein.pdf", "rb").read()
hb = open(f"{SRC}/handbuch.pdf", "rb").read()
PITCH_MM = 0.25


def radius_of(cells, shape="circle"):
    return next(r for r in wabe.RADII if len(wabe.build_lattice(r, shape).q) == len(cells))


def roundtrip(label, photo, truth):
    t = time.time()
    chunk, info = wabe.decode_image(photo)
    col = dicht.Collector()
    col.add(chunk)
    ok = col.result()[2] == truth
    print(f"  {label}: {'OK' if ok else 'ERROR'}  {info.note}  {time.time() - t:.1f}s")
    return info


print("1) Delivery note (3-page PDF, 5.2 KB) as circle, hexagon and 4-level circle")
c_circle = wabe.encode_file(ls, "lieferschein.pdf")[0]
c_hex = wabe.encode_file(ls, "lieferschein.pdf", shape="hexagon")[0]
c_gray = wabe.encode_file(ls, "lieferschein.pdf", levels=4)[0]
R1, Rh, Rg = radius_of(c_circle), radius_of(c_hex, "hexagon"), radius_of(c_gray)
print(f"  Circle R={R1} ({2 * R1 * PITCH_MM:.0f} mm), hexagon R={Rh}, 4 levels R={Rg} ({2 * Rg * PITCH_MM:.0f} mm)")
cv2.imwrite(f"{OUT}/lieferschein_kreis.png", wabe.render(c_circle, R1, ppm=6))
cv2.imwrite(f"{OUT}/lieferschein_sechseck.png", wabe.render(c_hex, Rh, "hexagon", ppm=6))
cv2.imwrite(f"{OUT}/lieferschein_4stufen.png", wabe.render(c_gray, Rg, ppm=6))
cv2.imwrite(f"{OUT}/lieferschein_punkte.png", wabe.render(c_circle, R1, ppm=6, style="dot"))

# Collage: three variants side by side
imgs = [cv2.imread(f"{OUT}/lieferschein_{n}.png", 0) for n in ("kreis", "sechseck", "4stufen")]
h = 560
imgs = [cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h), interpolation=cv2.INTER_AREA) for im in imgs]
coll = np.full((h + 40, sum(im.shape[1] for im in imgs) + 80, 3), 255, np.uint8)
x = 20
for im in imgs:
    coll[20:20 + h, x:x + im.shape[1]] = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    x += im.shape[1] + 20
cv2.imwrite(f"{OUT}/varianten.png", coll)

print("2) Simulated phone photo (skewed, lens, wave) with detection")
rgb = cv2.cvtColor(wabe.render(c_circle, R1, ppm=10), cv2.COLOR_GRAY2RGB)
photo = kamera.simulate_photo(rgb, 4.2, 10, seed=21, persp=0.13, angle_deg=35, k1=0.06, wave_amp=10, wave_len=1800)
info = roundtrip("Circle, 4.2 px/pitch", photo, ls)
cv2.imwrite(f"{OUT}/lieferschein_foto_erkannt.jpg", wabe.draw_debug(photo, info), [cv2.IMWRITE_JPEG_QUALITY, 88])
chunk, _ = wabe.decode_image(photo)
col = dicht.Collector(); col.add(chunk)
open(f"{OUT}/lieferschein_dekodiert.pdf", "wb").write(col.result()[2])

print("3) A4 page: manual PDF (16 pages, 42 KB) in one circle")
c_big = wabe.encode_file(hb, "handbuch.pdf")[0]
Rb = radius_of(c_big)
print(f"  R={Rb}, diameter {2 * Rb * PITCH_MM:.0f} mm, capacity {wabe.capacity(Rb)[2]} bytes")
big_png = f"{OUT}/handbuch_kreis.png"
cv2.imwrite(big_png, wabe.render(c_big, Rb, ppm=6))
side_mm = 2 * (Rb + 3) * PITCH_MM
pdf_path = f"{OUT}/handbuch_a4.pdf"
c = canvas.Canvas(pdf_path, pagesize=A4)
W, H = A4
c.setFont("Helvetica-Bold", 20)
c.drawString(20 * mm, H - 25 * mm, "wabe.")
c.setFont("Helvetica", 10)
c.drawString(20 * mm, H - 32 * mm, f"This page contains a 16-page PDF ({len(hb) // 1024} KB). Black/white, no server, no link.")
c.drawString(20 * mm, H - 37 * mm, f"Radius {Rb} cells on a hexagonal lattice, cell pitch {PITCH_MM} mm, 25 % Reed-Solomon.")
c.drawImage(big_png, (W - side_mm * mm) / 2, H - 45 * mm - side_mm * mm, side_mm * mm, side_mm * mm)
c.setFont("Helvetica", 8)
c.drawString(20 * mm, 15 * mm, "Prototype v0.1, gedacht.gemacht.")
c.save()
subprocess.run(["pdftoppm", "-png", "-r", "60", "-singlefile", pdf_path, f"{OUT}/handbuch_a4_vorschau"], check=True)
big_rgb = cv2.cvtColor(wabe.render(c_big, Rb, ppm=8), cv2.COLOR_GRAY2RGB)
# 12 MP photo of the A4 page: 210 mm over ~3000 px -> 14.3 px/mm * 0.25 mm = 3.6 px/pitch
roundtrip("A4 photo 12 MP, 3.6 px/pitch, lens + wave", kamera.simulate_photo(big_rgb, 3.6, 8, seed=31, k1=0.05, wave_amp=12, wave_len=2000), hb)

print("4) Screen: delivery note as 2 frames (R=64, 4 levels)")
frames = wabe.encode_file(ls, "lieferschein.pdf", R=64, levels=4)
ims = [Image.fromarray(wabe.render(f, 64, ppm=5, quiet=3)) for f in frames]
ims[0].save(f"{OUT}/lieferschein_screen.gif", save_all=True, append_images=ims[1:], duration=400, loop=0)
col = dicht.Collector()
for i, f in enumerate(frames):
    ph = kamera.simulate_photo(cv2.cvtColor(wabe.render(f, 64, ppm=10, quiet=3), cv2.COLOR_GRAY2RGB), 5.0, 10, seed=40 + i,
                               persp=0.05, angle_deg=5, jpeg_q=0, noise=4, light=(0.9, 1.0))
    chunk, info = wabe.decode_image(ph)
    col.add(chunk)
    print(f"  Frame {chunk.idx + 1}/{chunk.total}: {info.note}")
print(f"  {len(frames)} frames, reassembled identical={col.result()[2] == ls}")
print("done")
