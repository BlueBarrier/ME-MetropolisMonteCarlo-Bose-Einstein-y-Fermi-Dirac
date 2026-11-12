"""Utilidades compartidas por los scripts (rutas e impresión de tablas)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Consola de Windows: forzar UTF-8 para imprimir ±, σ, ε, etc.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--fast", action="store_true",
        help="Corridas cortas (5 %% de los pasos) en data/raw_fast y fig/fast, sólo para verificar.",
    )
    return parser


def print_table(rows: list[dict], columns: list[str], title: str | None = None) -> None:
    """Imprime una tabla de texto alineada."""
    if title:
        print(f"\n{title}")
    fmt_rows = [[_fmt(r.get(c, "")) for c in columns] for r in rows]
    widths = [max(len(c), *(len(fr[i]) for fr in fmt_rows)) if fmt_rows else len(c) for i, c in enumerate(columns)]
    print("  ".join(c.ljust(w) for c, w in zip(columns, widths)))
    print("  ".join("-" * w for w in widths))
    for fr in fmt_rows:
        print("  ".join(v.ljust(w) for v, w in zip(fr, widths)))


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4g}" if abs(v) < 1e4 else f"{v:.3e}"
    return str(v)
