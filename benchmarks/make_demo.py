"""Generate a controlled Poisson-blending comparison and metrics.

The case is deliberately synthetic: the intended in-scene appearance is known,
then a low-frequency exposure/white-balance mismatch is applied to make the
copy-paste source. This is a regression case, not a claim of general SOTA.
"""

import json
import subprocess
import time
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


def _smoothstep(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0.0, 1.0)
    return values * values * (3.0 - 2.0 * values)


def make_background(height: int = 480, width: int = 760) -> np.ndarray:
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    x = xx / float(width - 1)
    y = yy / float(height - 1)
    wall = np.stack(
        [
            0.55 + 0.19 * x + 0.06 * (1.0 - y),
            0.48 + 0.16 * x + 0.05 * (1.0 - y),
            0.38 + 0.10 * x + 0.04 * (1.0 - y),
        ],
        axis=2,
    )
    light = np.exp(-(((x - 0.71) / 0.23) ** 2 + ((y - 0.22) / 0.31) ** 2))
    wall += 0.12 * light[..., None]
    fine = 0.008 * np.sin(xx * 0.31) + 0.006 * np.sin(yy * 0.47 + xx * 0.07)
    wall += fine[..., None]

    table_line = 345
    table = np.stack(
        [0.31 + 0.13 * x, 0.24 + 0.10 * x, 0.17 + 0.07 * x], axis=2
    )
    wood_texture = 0.012 * np.sin(xx * 0.13 + yy * 0.03)
    table += wood_texture[..., None]
    transition = _smoothstep((yy - table_line + 2.0) / 4.0)[..., None]
    result = wall * (1.0 - transition) + table * transition
    return np.round(np.clip(result, 0.0, 1.0) * 255.0).astype(np.uint8)


def _rounded_box_sdf(
    xx: np.ndarray,
    yy: np.ndarray,
    center: Tuple[float, float],
    half_size: Tuple[float, float],
    radius: float,
) -> np.ndarray:
    qx = np.abs(xx - center[0]) - (half_size[0] - radius)
    qy = np.abs(yy - center[1]) - (half_size[1] - radius)
    outside = np.sqrt(np.maximum(qx, 0.0) ** 2 + np.maximum(qy, 0.0) ** 2)
    inside = np.minimum(np.maximum(qx, qy), 0.0)
    return outside + inside - radius


def make_product(height: int = 290, width: int = 270):
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    body_sdf = _rounded_box_sdf(xx, yy, (135.0, 180.0), (90.0, 105.0), 23.0)
    shoulder_sdf = _rounded_box_sdf(xx, yy, (135.0, 83.0), (57.0, 30.0), 12.0)
    cap_sdf = _rounded_box_sdf(xx, yy, (135.0, 39.0), (45.0, 27.0), 8.0)
    sdf = np.minimum(np.minimum(body_sdf, shoulder_sdf), cap_sdf)
    alpha = _smoothstep(0.5 - sdf / 2.2).astype(np.float32)

    # Intended target-scene appearance: a deep amber glass bottle.
    xnorm = (xx - 135.0) / 90.0
    ynorm = (yy - 180.0) / 105.0
    edge = np.clip(1.0 - np.abs(xnorm), 0.0, 1.0)
    illumination = 0.47 + 0.36 * edge + 0.12 * (1.0 - ynorm)
    true_rgb = np.zeros((height, width, 3), dtype=np.float32)
    true_rgb[..., 0] = 0.36 + 0.53 * illumination
    true_rgb[..., 1] = 0.12 + 0.27 * illumination
    true_rgb[..., 2] = 0.055 + 0.13 * illumination

    # Glass highlight and dark edge preserve high-frequency identity cues.
    highlight = np.exp(-((xx - 105.0) / 9.0) ** 2) * np.exp(-((yy - 165.0) / 95.0) ** 2)
    true_rgb += 0.22 * highlight[..., None]
    true_rgb *= (0.79 + 0.21 * edge[..., None])

    cap = cap_sdf <= 0.0
    cap_light = 0.55 + 0.30 * np.cos((xx - 135.0) / 45.0 * np.pi * 0.5)
    true_rgb[cap, 0] = cap_light[cap]
    true_rgb[cap, 1] = (0.42 + 0.25 * cap_light)[cap]
    true_rgb[cap, 2] = (0.17 + 0.12 * cap_light)[cap]

    # Draw a crisp label/logo so identity preservation is measurable.
    label = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(label)
    draw.rounded_rectangle((75, 151, 195, 220), radius=7, fill=(232, 213, 177, 242))
    draw.rectangle((75, 151, 195, 158), fill=(45, 33, 27, 245))
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 20)
        small = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 10)
    except OSError:
        font = ImageFont.load_default()
        small = font
    draw.text((91, 168), "PF / 01", font=font, fill=(43, 31, 25, 255))
    draw.text((98, 197), "POISSON FUSION", font=small, fill=(78, 54, 38, 255))
    label_rgba = np.asarray(label).astype(np.float32) / 255.0
    la = label_rgba[..., 3:4]
    true_rgb = true_rgb * (1.0 - la) + label_rgba[..., :3] * la

    # The copied source has an exposure/cyan-cast mismatch, not new texture.
    broad = 0.07 * np.cos((xx / width) * np.pi)[..., None]
    source_rgb = np.clip(
        true_rgb * np.array([0.78, 0.98, 1.23], dtype=np.float32)
        + np.array([0.025, 0.045, 0.075], dtype=np.float32)
        + broad,
        0.0,
        1.0,
    )
    true_rgb = np.clip(true_rgb, 0.0, 1.0)
    source_rgba = np.dstack([source_rgb, alpha])
    return (
        np.round(source_rgba * 255.0).astype(np.uint8),
        np.round(true_rgb * 255.0).astype(np.uint8),
        alpha,
    )


def alpha_paste(background, foreground_rgba, position):
    x, y = position
    h, w = foreground_rgba.shape[:2]
    output = background.astype(np.float32) / 255.0
    source = foreground_rgba[..., :3].astype(np.float32) / 255.0
    alpha = foreground_rgba[..., 3:4].astype(np.float32) / 255.0
    roi = output[y : y + h, x : x + w]
    output[y : y + h, x : x + w] = alpha * source + (1.0 - alpha) * roi
    return np.round(np.clip(output, 0.0, 1.0) * 255.0).astype(np.uint8)


def cv_clone(background, foreground_rgba, position, mode):
    x, y = position
    h, w = foreground_rgba.shape[:2]
    center = (x + w // 2, y + h // 2)
    source_bgr = cv2.cvtColor(foreground_rgba[..., :3], cv2.COLOR_RGB2BGR)
    target_bgr = cv2.cvtColor(background, cv2.COLOR_RGB2BGR)
    mask = np.where(foreground_rgba[..., 3] >= 8, 255, 0).astype(np.uint8)
    result = cv2.seamlessClone(source_bgr, target_bgr, mask, center, mode)
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)


def ground_truth(background, true_object, alpha, position):
    rgba = np.dstack([true_object, np.round(alpha * 255.0).astype(np.uint8)])
    return alpha_paste(background, rgba, position)


def srgb_to_linear(image):
    image = np.asarray(image, dtype=np.float32)
    return np.where(
        image <= 0.04045,
        image / 12.92,
        np.power((image + 0.055) / 1.055, 2.4),
    )


def linear_rgb_to_oklab(rgb):
    r, g, b = np.moveaxis(np.asarray(rgb, dtype=np.float32), -1, 0)
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l_, m_, s_ = np.cbrt(l), np.cbrt(m), np.cbrt(s)
    return np.stack(
        [
            0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
            1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
            0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
        ],
        axis=2,
    )


def _masked_psnr(result, truth, mask):
    values = (result.astype(np.float32) - truth.astype(np.float32))[mask]
    mse = float(np.mean(values * values)) if values.size else 0.0
    return 99.0 if mse <= 1.0e-12 else 10.0 * np.log10((255.0 * 255.0) / mse)


def _detail_correlation(result, truth, mask):
    result_y = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    truth_y = cv2.cvtColor(truth, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    result_hp = result_y - ndimage.gaussian_filter(result_y, 2.0)
    truth_hp = truth_y - ndimage.gaussian_filter(truth_y, 2.0)
    a = result_hp[mask]
    b = truth_hp[mask]
    if a.size < 8 or np.std(a) < 1.0e-8 or np.std(b) < 1.0e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _oklab_error(result, truth, mask):
    a = linear_rgb_to_oklab(srgb_to_linear(result.astype(np.float32) / 255.0))
    b = linear_rgb_to_oklab(srgb_to_linear(truth.astype(np.float32) / 255.0))
    return float(np.mean(np.linalg.norm(a[mask] - b[mask], axis=1)))


def metrics(result, truth, placed_alpha, runtime_ms):
    interior = placed_alpha >= 0.98
    boundary = (placed_alpha > 0.05) & (placed_alpha < 0.95)
    return {
        "runtime_ms": round(runtime_ms, 2),
        "interior_psnr_db": round(float(_masked_psnr(result, truth, interior)), 3),
        "boundary_psnr_db": round(float(_masked_psnr(result, truth, boundary)), 3),
        "detail_correlation": round(_detail_correlation(result, truth, interior), 5),
        "oklab_identity_error": round(_oklab_error(result, truth, interior), 6),
    }


def timed_call(function, repeats=5):
    """Return the last deterministic output and a small-sample median runtime."""

    function()  # warm-up: exclude one-time library and allocation setup
    durations = []
    output = None
    for _ in range(repeats):
        start = time.perf_counter()
        output = function()
        durations.append((time.perf_counter() - start) * 1000.0)
    return output, float(np.median(durations))


def rust_blend(position):
    binary = ROOT / "target" / "release" / "poissonfusion"
    if not binary.exists():
        subprocess.run(["cargo", "build", "--release"], cwd=ROOT, check=True)
    output = EXAMPLES / "poissonfusion_rust.png"
    diagnostics = ROOT / "benchmarks" / "rust_diagnostics.json"
    command = [
        str(binary),
        "--background",
        str(EXAMPLES / "background.png"),
        "--foreground",
        str(EXAMPLES / "foreground.png"),
        "--output",
        str(output),
        "--x",
        str(position[0]),
        "--y",
        str(position[1]),
        "--preset",
        "faithful",
        "--repeat",
        "5",
        "--diagnostics",
        str(diagnostics),
    ]
    subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    payload = json.loads(diagnostics.read_text(encoding="utf-8"))
    image = np.asarray(Image.open(output).convert("RGB"))
    return image, float(payload["runtime_ms_median"]), payload["solver"]


def comparison_grid(images: Dict[str, np.ndarray], destination: Path):
    panel_w, panel_h = 500, 316
    bar_h = 46
    columns = 3
    rows = 2
    canvas = Image.new("RGB", (panel_w * columns, (panel_h + bar_h) * rows), (18, 18, 18))
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    for index, (label, array) in enumerate(images.items()):
        row, column = divmod(index, columns)
        panel = Image.fromarray(array).resize((panel_w, panel_h), Image.Resampling.LANCZOS)
        x = column * panel_w
        y = row * (panel_h + bar_h)
        canvas.paste(panel, (x, y + bar_h))
        draw = ImageDraw.Draw(canvas)
        draw.text((x + 16, y + 12), label, font=font, fill=(242, 239, 230))
    canvas.save(destination, quality=95)


def main():
    EXAMPLES.mkdir(parents=True, exist_ok=True)
    background = make_background()
    source_rgba, true_object, alpha = make_product()
    position = (245, 82)
    truth = ground_truth(background, true_object, alpha, position)

    # Inputs are persisted before invoking the Rust executable.
    Image.fromarray(background).save(EXAMPLES / "background.png")
    Image.fromarray(source_rgba).save(EXAMPLES / "foreground.png")

    methods = {}
    timings = {}
    methods["Alpha paste"], timings["Alpha paste"] = timed_call(
        lambda: alpha_paste(background, source_rgba, position)
    )
    methods["OpenCV Normal Clone"], timings["OpenCV Normal Clone"] = timed_call(
        lambda: cv_clone(background, source_rgba, position, cv2.NORMAL_CLONE)
    )
    methods["OpenCV Mixed Clone"], timings["OpenCV Mixed Clone"] = timed_call(
        lambda: cv_clone(background, source_rgba, position, cv2.MIXED_CLONE)
    )
    rust_image, rust_runtime, rust_solver = rust_blend(position)
    methods["PoissonFusion Rust (ℓ=2)"] = rust_image
    timings["PoissonFusion Rust (ℓ=2)"] = rust_runtime

    placed_alpha = np.zeros(background.shape[:2], dtype=np.float32)
    h, w = alpha.shape
    x, y = position
    placed_alpha[y : y + h, x : x + w] = alpha
    report = {
        "scope": "controlled synthetic regression case; not a general SOTA benchmark",
        "image_size": list(background.shape[:2]),
        "runtime_note": "median of 5 warmed in-memory CPU calls on Apple M5; Rust and OpenCV paths measured independently",
        "poissonfusion_solver": rust_solver,
        "methods": {
            name: metrics(image, truth, placed_alpha, timings[name])
            for name, image in methods.items()
        },
    }

    Image.fromarray(np.round(alpha * 255.0).astype(np.uint8)).save(
        EXAMPLES / "mask.png"
    )
    Image.fromarray(truth).save(EXAMPLES / "synthetic_ground_truth.png")
    for name, image in methods.items():
        if name.startswith("PoissonFusion Rust"):
            # The Rust CLI already wrote the canonical output above.
            continue
        slug = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        Image.fromarray(image).save(EXAMPLES / (slug + ".png"))
    grid_images = {
        "Source mismatch (alpha paste)": methods["Alpha paste"],
        "Known synthetic ground truth": truth,
        "OpenCV Normal Clone": methods["OpenCV Normal Clone"],
        "OpenCV Mixed Clone": methods["OpenCV Mixed Clone"],
        "PoissonFusion Rust — ℓ=2": methods["PoissonFusion Rust (ℓ=2)"],
        "Input background": background,
    }
    comparison_grid(grid_images, EXAMPLES / "comparison.png")
    (ROOT / "benchmarks" / "results.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
