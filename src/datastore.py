"""Datos crudos: guardado y lectura con metadatos (semilla, parámetros y versiones).

Cada archivo ``.npz`` guarda los arreglos numéricos más una entrada ``metadata``
con un JSON que incluye la semilla, los parámetros de la corrida, la fecha, el
commit de git y las versiones de Python y de las bibliotecas. Así cada figura
se puede rastrear hasta la corrida que la produjo.
"""

from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
FIG_DIR = ROOT / "figures"


def environment_metadata() -> dict:
    """Versiones del entorno y commit actual (para reproducibilidad)."""
    import matplotlib
    import numba
    import scipy

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=5
        ).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        commit = None
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "numba": numba.__version__,
        "matplotlib": matplotlib.__version__,
        "git_commit": commit,
    }


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def save_raw(path: str | Path, arrays: dict, metadata: dict) -> Path:
    """Guarda ``arrays`` (dict de ndarrays) y ``metadata`` (dict) en un .npz comprimido."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"environment": environment_metadata(), **_jsonable(metadata)}
    np.savez_compressed(path, metadata=np.array(json.dumps(meta, ensure_ascii=False)), **arrays)
    return path


def load_raw(path: str | Path) -> tuple[dict, dict]:
    """Lee un .npz escrito con ``save_raw``. Devuelve (arreglos, metadatos)."""
    with np.load(Path(path), allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files if k != "metadata"}
        meta = json.loads(str(data["metadata"])) if "metadata" in data.files else {}
    return arrays, meta


def write_table_csv(path: str | Path, rows: list[dict]) -> Path:
    """Tabula una lista de diccionarios en CSV (UTF-8)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    fields = list(rows[0].keys())
    for r in rows[1:]:
        fields += [k for k in r if k not in fields]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: _fmt(r.get(k, "")) for k in fields})
    return path


def read_table_csv(path: str | Path) -> list[dict]:
    """Lee un CSV escrito con ``write_table_csv``, convirtiendo números cuando se puede."""
    with Path(path).open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for r in rows:
        conv = {}
        for k, v in r.items():
            try:
                conv[k] = int(v)
            except ValueError:
                try:
                    conv[k] = float(v)
                except ValueError:
                    conv[k] = v
        out.append(conv)
    return out


def _fmt(v):
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.10g}"
    if isinstance(v, np.generic):
        return v.item()
    return v
