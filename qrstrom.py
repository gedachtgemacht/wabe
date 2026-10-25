"""QR variant of the stream for direct comparison: same fountain code, same frame logic,
but QR version 40 (177 modules) instead of the wabe. circle. Encoder: qrcode, decoder: zxing-cpp (C++)."""
import numpy as np
import cv2
import qrcode
import zxingcpp

import strom

QR_CAP = {"L": 2953, "M": 2331, "Q": 1663, "H": 1273}            # byte mode, version 40
QR_EC = {"L": qrcode.constants.ERROR_CORRECT_L, "M": qrcode.constants.ERROR_CORRECT_M,
         "Q": qrcode.constants.ERROR_CORRECT_Q, "H": qrcode.constants.ERROR_CORRECT_H}


def qr_png(payload: bytes, ec: str = "L", box: int = 5) -> bytes:
    q = qrcode.QRCode(version=40, error_correction=QR_EC[ec], box_size=box, border=4)
    q.add_data(qrcode.util.QRData(payload, mode=qrcode.util.MODE_8BIT_BYTE))
    q.make(fit=False)
    im = np.array(q.make_image(fill_color="black", back_color="white").convert("L"))
    return cv2.imencode(".png", im)[1].tobytes()


def qr_stream(data: bytes, name: str, n_frames: int = 0, ec: str = "L", box: int = 5):
    enc = strom.StreamEncoder(data, name, payload_cap=QR_CAP[ec])
    n = n_frames or 4 * enc.K
    pngs = [qr_png(enc.frame_payload(i + 1), ec, box) for i in range(n)]
    return enc, pngs


def decode_qr_payloads(img) -> list[bytes]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return [r.bytes for r in zxingcpp.read_barcodes(gray, formats=zxingcpp.BarcodeFormat.QRCode) if r.bytes]
