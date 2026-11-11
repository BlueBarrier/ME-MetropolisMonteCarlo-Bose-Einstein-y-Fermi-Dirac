"""Digitalización de datos experimentales publicados (figuras en PDF → CSV).

Procedimiento, equivalente a usar WebPlotDigitizer pero automático y reproducible:
  1. La página del PDF se renderiza a 600 dpi con ``pdftoppm`` (Poppler) y se recorta la figura.
  2. Se detecta el marco de los ejes (filas/columnas con trazos oscuros largos) y las marcas
     ("ticks") principales. La calibración píxel → dato es un ajuste lineal sobre las marcas,
     y su residuo cuantifica el error de calibración.
  3. Los marcadores rellenos se aíslan con una apertura morfológica (borra las líneas
     delgadas: curvas teóricas, barras de error, texto fino). Las manchas de área k veces la
     de un marcador se separan en k puntos con k-medias.
  4. Se escribe un CSV con los puntos (más una incertidumbre de digitalización) y una imagen
     de control con los puntos detectados superpuestos.

Los PDF tienen derechos de autor y no se incluyen en el repositorio. Para repetir la
digitalización hay que obtenerlos (ver ``data/external/README.md``) y correr:

    python scripts/digitize_external.py --demarco RUTA/DeMarco_Jin_1999.pdf --cornell RUTA/cond-mat_9903109.pdf

Salidas: data/external/*.csv y data/external/check_*.png
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import _common  # noqa: F401
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.cluster.vq import kmeans2

from src.datastore import ROOT

EXT_DIR = ROOT / "data" / "external"
DPI = 600


@dataclass
class FigureSpec:
    name: str
    page: int
    crop: tuple[int, int, int, int]  # (x0, y0, x1, y1) en píxeles a 600 dpi
    x_ticks: tuple[float, float]  # (valor de la primera marca principal interior desde la izquierda, paso)
    y_ticks: tuple[float, float]  # (valor de la primera marca principal interior desde abajo, paso)
    y_label: str
    x_label: str
    exclude: list[tuple[float, float, float, float]] = field(default_factory=list)  # cajas en datos
    marker_open_radius: int = 9
    x_range: tuple[float, float] | None = None  # recorte final en x (datos)
    error_bar_x: float | None = None  # posición aproximada de una barra de error "típica" a medir
    note: str = ""


def render_page(pdf: Path, page: int, out_dir: Path) -> Image.Image:
    prefix = out_dir / f"p{page}"
    subprocess.run(["pdftoppm", "-r", str(DPI), "-f", str(page), "-l", str(page), "-png", "-gray",
                    str(pdf), str(prefix)], check=True, capture_output=True)
    png = sorted(out_dir.glob(f"p{page}*.png"))[-1]
    return Image.open(png).convert("L")


def find_frame(dark: np.ndarray) -> tuple[int, int, int, int]:
    """(izquierda, derecha, arriba, abajo) del marco: trazos continuos más largos."""
    rows = dark.sum(axis=1)
    cols = dark.sum(axis=0)
    h_lines = np.nonzero(rows > 0.5 * dark.shape[1])[0]
    v_lines = np.nonzero(cols > 0.5 * dark.shape[0])[0]
    top, bottom = h_lines.min(), h_lines.max()
    left, right = v_lines.min(), v_lines.max()
    return int(left), int(right), int(top), int(bottom)


def _runs(mask_1d: np.ndarray) -> list[tuple[int, int]]:
    idx = np.nonzero(mask_1d)[0]
    if idx.size == 0:
        return []
    splits = np.nonzero(np.diff(idx) > 1)[0]
    starts = np.r_[idx[0], idx[splits + 1]]
    ends = np.r_[idx[splits], idx[-1]]
    return list(zip(starts, ends))


def _tick_lengths(band: np.ndarray) -> np.ndarray:
    """Para cada fila de ``band``: largo del trazo oscuro que empieza en la columna 0."""
    lengths = np.zeros(band.shape[0], dtype=int)
    for r, row in enumerate(band):
        if row[0]:
            stop = np.nonzero(~row)[0]
            lengths[r] = stop[0] if stop.size else row.size
    return lengths


def find_major_ticks(dark: np.ndarray, frame: tuple[int, int, int, int], thick: int, axis: str) -> np.ndarray:
    """Posiciones (en píxeles) de las marcas principales interiores del eje ``axis`` ("x" o "y").

    Las marcas salen hacia dentro desde el borde inferior (eje x) o izquierdo (eje y).
    Se distinguen de las secundarias por su largo.
    """
    left, right, top, bottom = frame
    g = thick + 2
    if axis == "y":
        band = dark[top + g: bottom - g, left + g: left + g + 80]
        offset = top + g
    else:  # columnas de abajo hacia arriba, como filas
        band = dark[bottom - g - 80: bottom - g, left + g: right - g][::-1].T
        offset = left + g
    lengths = _tick_lengths(band)
    centers, lens = [], []
    for a, b in _runs(lengths > 0):
        centers.append(offset + 0.5 * (a + b))
        lens.append(lengths[a: b + 1].max())
    centers, lens = np.array(centers), np.array(lens)
    return np.sort(centers[lens >= 0.75 * lens.max()])


def calibrate(major_px: np.ndarray, first: float, step: float, axis: str) -> tuple[np.poly1d, float]:
    """Ajuste lineal píxel → valor sobre las marcas principales (y: de abajo hacia arriba)."""
    px = major_px[::-1] if axis == "y" else major_px
    vals = first + step * np.arange(px.size)
    p = np.poly1d(np.polyfit(px, vals, 1))
    return p, float(np.max(np.abs(p(px) - vals)))


def detect_markers(dark: np.ndarray, radius: int) -> list[tuple[float, float]]:
    """Centros (columna, fila) de los marcadores rellenos."""
    yy, xx = np.mgrid[-radius: radius + 1, -radius: radius + 1]
    disk = xx**2 + yy**2 <= radius**2
    opened = ndimage.binary_opening(dark, structure=disk)
    labels, n = ndimage.label(opened)
    if n == 0:
        return []
    areas = ndimage.sum(opened, labels, index=np.arange(1, n + 1))
    unit = np.median(areas[areas > 0.3 * np.median(areas)])
    points = []
    for k in range(1, n + 1):
        a = areas[k - 1]
        if a < 0.35 * unit:
            continue
        ys, xs = np.nonzero(labels == k)
        count = max(1, int(round(a / unit)))
        if count == 1:
            points.append((xs.mean(), ys.mean()))
        else:
            pts = np.column_stack([xs, ys]).astype(float)
            centers, _ = kmeans2(pts, count, seed=1, minit="++")
            points += [tuple(c) for c in centers]
    return points


def digitize(img: Image.Image, spec: FigureSpec, out_png: Path) -> tuple[np.ndarray, dict]:
    crop = img.crop(spec.crop)
    a = np.asarray(crop)
    dark = a < 128
    left, right, top, bottom = find_frame(dark)
    thick = int(np.sum(dark[(top + bottom) // 2, left: left + 20]))  # grosor del marco
    frame = (left, right, top, bottom)
    major_x = find_major_ticks(dark, frame, thick, "x")
    major = find_major_ticks(dark, frame, thick, "y")
    x_of, x_resid = calibrate(major_x, *spec.x_ticks, "x")
    y_of, y_resid = calibrate(major, *spec.y_ticks, "y")
    inner = np.zeros_like(dark)
    m = thick + 3
    inner[top + m: bottom - m, left + m: right - m] = dark[top + m: bottom - m, left + m: right - m]
    pts = detect_markers(inner, spec.marker_open_radius)
    data = []
    for cx, cy in pts:
        x, y = float(x_of(cx)), float(y_of(cy))
        if any(x0 <= x <= x1 and y0 <= y <= y1 for x0, x1, y0, y1 in spec.exclude):
            continue
        if spec.x_range and not (spec.x_range[0] <= x <= spec.x_range[1]):
            continue
        data.append((x, y, cx, cy))
    data = np.array(sorted(data))
    px_x = abs(spec.x_ticks[1] / np.median(np.diff(major_x)))
    px_y = abs(spec.y_ticks[1] / np.median(np.diff(major)))
    info = {
        "frame_px": frame, "major_x_ticks_px": major_x.tolist(), "major_y_ticks_px": major.tolist(),
        "x_calibration_max_residual": x_resid, "y_calibration_max_residual": y_resid, "data_per_px_x": px_x, "data_per_px_y": px_y,
        "n_points": int(len(data)), "sigma_y": None,
    }
    if spec.error_bar_x is not None:
        # Barra de error "típica": trazo vertical continuo que atraviesa el marcador más cercano
        # a x = error_bar_x. Se recorre la columna del centro del marcador hacia arriba y hacia abajo.
        k = int(np.argmin(np.abs(data[:, 0] - spec.error_bar_x)))
        cx, cy = int(round(data[k, 2])), int(round(data[k, 3]))
        up = cy
        while up > 0 and inner[up - 1, cx]:
            up -= 1
        down = cy
        while down < inner.shape[0] - 1 and inner[down + 1, cx]:
            down += 1
        half_up, half_down = y_of(up) - y_of(cy), y_of(cy) - y_of(down)
        info["sigma_y"] = float(0.5 * (half_up + half_down))
        info["sigma_y_halves"] = (float(half_up), float(half_down))
    # Imagen de control: puntos detectados en rojo sobre la figura original.
    check = crop.convert("RGB")
    draw = ImageDraw.Draw(check)
    for _, _, cx, cy in data:
        draw.ellipse((cx - 14, cy - 14, cx + 14, cy + 14), outline=(220, 30, 30), width=5)
    for r in major:
        draw.line((left, r, left + 80, r), fill=(30, 120, 220), width=3)
    for c in major_x:
        draw.line((c, bottom, c, bottom - 80), fill=(30, 120, 220), width=3)
    check.save(out_png)
    return data[:, :2], info


SPECS = {
    "demarco": FigureSpec(
        name="demarco_jin_1999_fig4",
        page=3,
        crop=(440, 3990, 1900, 5080),
        x_ticks=(0.5, 0.5),
        y_ticks=(0.00, 0.05),
        x_label="T/T_F",
        y_label="dU/U_cl",
        x_range=(0.0, 3.3),
        error_bar_x=2.45,
        note="DeMarco y Jin, Science 285, 1703 (1999), Fig. 4: exceso de energía dU/U_cl = (U - 3NkT)/3NkT "
             "vs T/T_F, con T_F = hbar(6 w_z w_r^2 N)^(1/3)/k. Barra de error típica (estadística) indicada en la figura.",
    ),
    "cornell": FigureSpec(
        name="ensher_1996_energy_via_cornell_review_fig5",
        page=23,
        crop=(1300, 1150, 4500, 4200),
        x_ticks=(0.6, 0.2),
        y_ticks=(0.5, 0.5),
        x_label="T/T_0",
        y_label="E_rel/(N k T_0)",
        # recuadro interior (inset) en coordenadas de datos: se excluye
        exclude=[(0.92, 1.80, -0.05, 1.03)],
        note="Ensher et al., PRL 77, 4984 (1996), reproducida en Cornell, Ensher y Wieman, cond-mat/9903109, Fig. 5: "
             "energía de liberación (cinética + interacción) por partícula en unidades de k_B T_0 vs T/T_0(N). "
             "La recta clásica es 1.5 T/T_0.",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--demarco", type=Path, help="PDF de DeMarco y Jin, Science 285, 1703 (1999)")
    parser.add_argument("--cornell", type=Path, help="PDF de arXiv:cond-mat/9903109 (Cornell, Ensher, Wieman)")
    args = parser.parse_args()
    EXT_DIR.mkdir(parents=True, exist_ok=True)
    for key, pdf in (("demarco", args.demarco), ("cornell", args.cornell)):
        if pdf is None:
            continue
        spec = SPECS[key]
        with tempfile.TemporaryDirectory() as tmp:
            img = render_page(pdf, spec.page, Path(tmp))
            data, info = digitize(img, spec, EXT_DIR / f"check_{spec.name}.png")
        out = EXT_DIR / f"{spec.name}.csv"
        with out.open("w", encoding="utf-8") as fh:
            fh.write(f"# {spec.note}\n")
            fh.write(f"# Digitalizado con scripts/digitize_external.py a {DPI} dpi. Resolución: "
                     f"{info['data_per_px_x']:.2e} (x) y {info['data_per_px_y']:.2e} (y) por píxel; "
                     f"residuos máximos de la calibración: {info['x_calibration_max_residual']:.2e} (x), "
                     f"{info['y_calibration_max_residual']:.2e} (y).\n")
            if info["sigma_y"] is not None:
                fh.write(f"# sigma_y = semilongitud de la barra de error típica de la figura = {info['sigma_y']:.4f}\n")
            fh.write(f"{spec.x_label},{spec.y_label}\n")
            for x, y in data:
                fh.write(f"{x:.4f},{y:.5f}\n")
        print(f"{spec.name}: {info['n_points']} puntos -> {out}")
        print(f"  marco {info['frame_px']}, marcas x {np.round(info['major_x_ticks_px'], 1)}, "
              f"marcas y {np.round(info['major_y_ticks_px'], 1)}")
        print(f"  residuos de calibración: x {info['x_calibration_max_residual']:.2e}, "
              f"y {info['y_calibration_max_residual']:.2e}")


if __name__ == "__main__":
    main()
