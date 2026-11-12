"""Visor de figuras: muestra los PNG uno por uno en una ventana hasta que el usuario la cierra.

Controles:
  → , espacio, n, botón "Siguiente"   : figura siguiente
  ← , retroceso, p, botón "Anterior"  : figura anterior
  Inicio / Fin                         : primera / última
  cerrar la ventana (o tecla q)        : terminar
"""

from __future__ import annotations

from pathlib import Path


def show_gallery(paths, title: str = "Figuras del proyecto") -> bool:
    """Abre la ventana con las imágenes de ``paths`` (sólo .png). Devuelve False si no hay pantalla."""
    images = [Path(p) for p in paths if Path(p).suffix.lower() == ".png" and Path(p).exists()]
    if not images:
        print("No hay imágenes PNG para mostrar.")
        return False

    import matplotlib
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button

    try:
        plt.switch_backend("TkAgg")  # ventana interactiva (Tk viene con Python)
    except Exception as exc:  # p. ej. servidor sin pantalla
        print(f"No se pudo abrir una ventana ({exc}). Las figuras están en {images[0].parent}.")
        return False

    with matplotlib.rc_context(rc=matplotlib.rcParamsDefault):
        fig = plt.figure(figsize=(12, 7.8))
        try:
            fig.canvas.manager.set_window_title(title)
        except AttributeError:
            pass
        ax = fig.add_axes((0.01, 0.10, 0.98, 0.84))
        ax_prev = fig.add_axes((0.30, 0.02, 0.15, 0.055))
        ax_next = fig.add_axes((0.55, 0.02, 0.15, 0.055))
        b_prev, b_next = Button(ax_prev, "◀  Anterior"), Button(ax_next, "Siguiente  ▶")
        state = {"i": 0}

        def draw():
            path = images[state["i"]]
            ax.clear()
            ax.imshow(mpimg.imread(path))
            ax.set_axis_off()
            pretty = path.stem.replace("_", " ")
            ax.set_title(f"{state['i'] + 1}/{len(images)}   {pretty}", fontsize=12, loc="center")
            fig.canvas.draw_idle()

        def move(step):
            state["i"] = (state["i"] + step) % len(images)
            draw()

        def on_key(event):
            if event.key in ("right", " ", "n", "down", "pagedown"):
                move(+1)
            elif event.key in ("left", "backspace", "p", "up", "pageup"):
                move(-1)
            elif event.key == "home":
                state["i"] = 0
                draw()
            elif event.key == "end":
                state["i"] = len(images) - 1
                draw()

        b_prev.on_clicked(lambda _: move(-1))
        b_next.on_clicked(lambda _: move(+1))
        fig.canvas.mpl_connect("key_press_event", on_key)
        draw()
        print(f"Mostrando {len(images)} figuras. Use ← / → o los botones; cierre la ventana para terminar.")
        plt.show()  # bloquea hasta que el usuario cierra la ventana
    return True
