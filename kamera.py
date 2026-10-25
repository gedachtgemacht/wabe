"""Camera/scanner simulation for dicht. tests: perspective, rotation, blur,
light gradient, colour cast, noise, JPEG with chroma subsampling."""
import math
import numpy as np
import cv2


def simulate_photo(code_rgb: np.ndarray, target_ppm: float, src_ppm: int, seed: int = 1,
                   persp: float = 0.10, angle_deg: float = 12.0, blur_sigma: float = 0.9,
                   noise: float = 5.0, jpeg_q: int = 72, light=(0.72, 1.08), cast=(0.93, 1.0, 1.05),
                   paper: int = 236, k1: float = 0.0, wave_amp: float = 0.0, wave_len: float = 600.0) -> np.ndarray:
    """code_rgb: rendered code image (RGB, src_ppm pixels/module). Returns a BGR 'photo' with ~target_ppm."""
    rng = np.random.default_rng(seed)
    h, w = code_rgb.shape[:2]
    # Paper/background with slight texture
    ph, pw = int(h * 1.35), int(w * 1.35)
    canvas = np.full((ph, pw, 3), paper, np.float32)
    canvas += rng.normal(0, 2.5, (ph, pw, 1)).astype(np.float32)
    oy, ox = (ph - h) // 2, (pw - w) // 2
    canvas[oy:oy + h, ox:ox + w] = code_rgb.astype(np.float32)

    scale = target_ppm / src_ppm
    src = np.array([[0, 0], [pw, 0], [pw, ph], [0, ph]], np.float32)
    jitter = rng.uniform(-persp, persp, (4, 2)).astype(np.float32) * np.array([pw, ph], np.float32)
    dst = (src + jitter) * scale
    P = cv2.getPerspectiveTransform(src, dst)
    ang = np.deg2rad(rng.uniform(-angle_deg, angle_deg))
    cx, cy = pw * scale / 2, ph * scale / 2
    R = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]], np.float64)
    T1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], np.float64)
    T2 = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]], np.float64)
    M = T2 @ R @ T1 @ P
    corners = cv2.perspectiveTransform(src.reshape(-1, 1, 2), M).reshape(-1, 2)
    mn = corners.min(0)
    M = np.array([[1, 0, -mn[0]], [0, 1, -mn[1]], [0, 0, 1]]) @ M
    size = np.ceil(corners.max(0) - mn).astype(int)
    out = cv2.warpPerspective(canvas, M, (int(size[0]), int(size[1])), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(paper * 0.8,) * 3)
    # Lens distortion (radial) and wavy paper (sine displacement)
    H, W = out.shape[:2]
    if k1 or wave_amp:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        cx, cy = W / 2.0, H / 2.0
        xn, yn = (xx - cx) / (W / 2.0), (yy - cy) / (H / 2.0)
        r2 = xn * xn + yn * yn
        f = 1.0 + k1 * r2 + k1 * 0.3 * r2 * r2
        mx = cx + (xx - cx) * f
        my = cy + (yy - cy) * f
        if wave_amp:
            mx = mx + wave_amp * np.sin(2 * np.pi * yy / wave_len)
            my = my + wave_amp * 0.6 * np.cos(2 * np.pi * xx / (wave_len * 1.3))
        out = cv2.remap(out, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=(paper * 0.8,) * 3)
    # Light: gradient + vignette
    gx = np.linspace(light[0], light[1], W, dtype=np.float32)[None, :]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    vign = 1.0 - 0.18 * (((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    out = out * (gx * vign)[..., None]
    out = out * np.array(cast, np.float32)[None, None, :]        # RGB colour cast
    if blur_sigma > 0:
        out = cv2.GaussianBlur(out, (0, 0), blur_sigma)
    out = out + rng.normal(0, noise, out.shape).astype(np.float32)
    out = np.clip(out, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    if jpeg_q:
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_q,
                                             cv2.IMWRITE_JPEG_SAMPLING_FACTOR, 0x221111])  # 4:2:0
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return bgr


def simulate_scan(code_rgb: np.ndarray, target_ppm: float, src_ppm: int, seed: int = 1,
                  blur_sigma: float = 0.7, noise: float = 3.0, angle_deg: float = 1.5) -> np.ndarray:
    """Flatbed scanner: almost no perspective, slight rotation, little noise, no JPEG."""
    return simulate_photo(code_rgb, target_ppm, src_ppm, seed, persp=0.004, angle_deg=angle_deg,
                          blur_sigma=blur_sigma, noise=noise, jpeg_q=0, light=(0.97, 1.0), cast=(1, 1, 1))


def simulate_screen_photo(screen_rgb: np.ndarray, cam_per_screen: float = 1.8, seed: int = 1,
                          gap: float = 0.3, persp: float = 0.05, angle_deg: float = 4.0,
                          blur_sigma: float = 0.8, noise: float = 4.0, jpeg_q: int = 80,
                          glare: float = 0.25) -> np.ndarray:
    """Photo of a screen: pixel grid with dark gaps (together with perspective
    and blur this produces the typical moire), slight perspective, glare, JPEG."""
    rng = np.random.default_rng(seed)
    h, w = screen_rgb.shape[:2]
    f = int(math.ceil(cam_per_screen * 2))                     # upscale, then pixel grid
    big = np.repeat(np.repeat(screen_rgb.astype(np.float32), f, 0), f, 1)
    mask = np.ones((h * f, w * f, 1), np.float32)
    mask[f - 1::f, :, :] *= (1.0 - gap)                       # row gap
    mask[:, f - 1::f, :] *= (1.0 - gap)                       # column gap
    big = big * mask
    # Glare: wide bright spot
    yy, xx = np.mgrid[0:h * f, 0:w * f].astype(np.float32)
    cx, cy = rng.uniform(0.2, 0.8) * w * f, rng.uniform(0.2, 0.8) * h * f
    g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (0.35 * w * f) ** 2))
    big = big + glare * 255 * g[..., None]
    big = np.clip(big, 0, 255).astype(np.uint8)
    return simulate_photo(big, cam_per_screen, f, seed, persp=persp, angle_deg=angle_deg, blur_sigma=blur_sigma,
                          noise=noise, jpeg_q=jpeg_q, light=(0.9, 1.0), cast=(1.0, 1.0, 1.02), paper=30)
