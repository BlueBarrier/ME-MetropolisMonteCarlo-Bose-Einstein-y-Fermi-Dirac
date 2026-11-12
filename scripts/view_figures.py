"""Abre una ventana para ver, una por una, las figuras PNG guardadas en fig/.

Uso: python scripts/view_figures.py [--fast]
     (← / → o los botones para avanzar; cierre la ventana para terminar)
"""

from __future__ import annotations

import _common

from src.config import output_dirs
from src.viewer import show_gallery


def main(fast: bool = False) -> None:
    _, fig_dir = output_dirs(fast)
    show_gallery(sorted(fig_dir.glob("*.png")))


if __name__ == "__main__":
    args = _common.base_parser(__doc__.splitlines()[0]).parse_args()
    main(fast=args.fast)
