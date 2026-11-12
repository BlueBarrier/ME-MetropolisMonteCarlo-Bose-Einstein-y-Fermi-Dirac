"""Análisis estadístico de series temporales correlacionadas (salidas de MCMC).

Las mediciones consecutivas de una cadena de Markov están correlacionadas, así
que el error estándar ingenuo σ_A/√N subestima la incertidumbre. Aquí se usa

    σ²_Ā ≃ (2 τ_int / N) σ²_A,     τ_int = 1/2 + Σ_{t≥1} C(t),
    C(t) = [⟨A_k A_{k+t}⟩ - ⟨A⟩²] / σ²_A,

con la suma cortada por una ventana automática (Sokal) o en el primer C(t) ≤ 0.
"""

from __future__ import annotations

import numpy as np


def autocorrelation_function(series: np.ndarray, max_lag: int | None = None) -> np.ndarray:
    """C(t) normalizada, con C(0) = 1, calculada con FFT.

    Se usa el estimador ⟨A_k A_{k+t}⟩ promediado sobre los n - t pares disponibles.
    Si la serie es constante devuelve C = [1, 0, 0, ...].
    """
    x = np.asarray(series, dtype=float)
    n = x.size
    max_lag = n - 1 if max_lag is None else min(max_lag, n - 1)
    x = x - x.mean()
    var = np.dot(x, x) / n
    if n < 2 or var == 0.0:
        out = np.zeros(max_lag + 1)
        out[0] = 1.0
        return out
    nfft = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(x, n=nfft)
    acov = np.fft.irfft(f * np.conj(f), n=nfft)[: max_lag + 1]
    acov /= n - np.arange(max_lag + 1)  # promedio sobre pares disponibles
    return acov / var


def integrated_autocorrelation_time(
    series: np.ndarray, method: str = "sokal", c: float = 5.0
) -> float:
    """τ_int en unidades del intervalo de muestreo.

    method="sokal"      : ventana W mínima tal que W ≥ c·τ_int(W) (Madras y Sokal 1988).
    method="first_zero" : suma C(t) hasta antes del primer t con C(t) ≤ 0.

    Se devuelve como mínimo 1/2, el valor de una serie sin correlación.
    """
    x = np.asarray(series, dtype=float)
    if x.size < 4 or np.all(x == x[0]):
        return 0.5
    C = autocorrelation_function(x, max_lag=x.size // 2)
    if method == "first_zero":
        nonpos = np.nonzero(C[1:] <= 0.0)[0]
        cut = nonpos[0] + 1 if nonpos.size else C.size
        tau = 0.5 + C[1:cut].sum()
    elif method == "sokal":
        tau_w = 0.5 + np.cumsum(C[1:])
        W = np.arange(1, C.size)
        ok = np.nonzero(W >= c * tau_w)[0]
        tau = tau_w[ok[0]] if ok.size else tau_w[-1]
    else:
        raise ValueError("method debe ser 'sokal' o 'first_zero'")
    return float(max(tau, 0.5))


def statistical_uncertainty(
    series: np.ndarray, method: str = "sokal", resolution: float = 0.0
) -> tuple[float, float, float]:
    """(promedio, τ_int, error estándar corregido) con σ_Ā = √(2 τ_int σ²_A / N).

    ``resolution`` es el salto mínimo que puede dar el observable (p. ej. 1/N para
    n_0/N). Si la serie no fluctuó en toda la corrida (σ_A = 0, como n_0 = 1 en
    fermiones a T baja), el error no es cero sino desconocido. En ese caso se usa
    la cota "regla de tres": una fracción de tiempo ≤ 3/n fuera del valor observado
    (95 %), es decir σ = resolution·3/n.
    """
    x = np.asarray(series, dtype=float)
    n = x.size
    mean = float(x.mean())
    if n < 2:
        return mean, 0.5, float("nan")
    if np.all(x == x[0]):  # se compara la serie, no var == 0: el redondeo deja var ~ 1e-35
        return float(x[0]), 0.5, resolution * 3.0 / n
    tau = integrated_autocorrelation_time(x, method=method)
    var = float(x.var(ddof=1))
    return mean, tau, float(np.sqrt(2.0 * tau * var / n))


def discrepancy_in_sigmas(sim_mean: float, sim_err: float, exact_val: float) -> float:
    """z = |Ā_sim - A_exacto| / σ_Ā. z < 2 se considera compatible (≈95 %)."""
    diff = abs(sim_mean - exact_val)
    if not np.isfinite(sim_err) or sim_err <= 0.0:
        return 0.0 if np.isclose(diff, 0.0, atol=1e-12) else float("inf")
    return float(diff / sim_err)


def moving_average(series: np.ndarray, window: int) -> np.ndarray:
    """Promedio móvil (sólo la parte válida, longitud n - window + 1)."""
    x = np.asarray(series, dtype=float)
    window = max(1, min(window, x.size))
    c = np.cumsum(np.insert(x, 0, 0.0))
    return (c[window:] - c[:-window]) / window


def estimate_burn_in(trace_cold: np.ndarray, trace_hot: np.ndarray, window: int = 20) -> int | None:
    """Índice en que se cruzan los promedios móviles de dos cadenas.

    Una cadena arranca del estado base ("fría") y otra de una configuración
    aleatoria ("caliente"). Las dos convergen al mismo valor estacionario desde lados
    opuestos, y el primer cruce de sus promedios móviles es una cota del tiempo de
    equilibración. Devuelve el índice (en muestras) o None si no se cruzan.
    """
    a = moving_average(trace_cold, window)
    b = moving_average(trace_hot, window)
    m = min(a.size, b.size)
    diff = b[:m] - a[:m]
    sign0 = np.sign(diff[0])
    crossed = np.nonzero(np.sign(diff) != sign0)[0]
    if sign0 == 0:
        return 0
    return int(crossed[0] + window - 1) if crossed.size else None
