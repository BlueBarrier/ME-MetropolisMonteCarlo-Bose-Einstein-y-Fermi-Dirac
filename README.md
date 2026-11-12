# Estadística cuántica por Monte Carlo: condensación de Bose–Einstein vs. mar de Fermi

Proyecto final de **Mecánica Estadística** (Universidad del Valle de Guatemala).

Simulación de gases ideales de **bosones** y **fermiones** (una componente de espín) en una
trampa armónica isotrópica 3D, en el **ensemble canónico** ($N$, $T$, $\omega$ fijos), con
**Markov Chain Monte Carlo de Metropolis** sobre configuraciones de ocupación. Se valida contra
resultados exactos y se estudia cómo las restricciones de ocupación producen la acumulación
de bosones en el estado fundamental y el llenado del mar de Fermi, en función de $T$ y $N$.
También se comparan los resultados con las distribuciones de Bose–Einstein y Fermi–Dirac
del ensemble gran canónico.

**Integrantes:** Isabella Ramírez, Erick Barrera, José Dávila.

---

## Modelo en una línea

$$
H(\mathbf n)=\sum_i n_i\,\varepsilon_i,\quad \varepsilon_i=\hbar\omega(q_x+q_y+q_z),\quad \sum_i n_i=N,\quad
n_i\in\{0,\dots,N\}\ \text{(BE)},\ \{0,1\}\ \text{(FD)},\quad P(\mathbf n)\propto e^{-\beta H(\mathbf n)}
$$

Unidades reducidas: **ħω = k_B = 1** (energía en ħω, temperatura en ħω/k_B). Las conversiones
al SI están en `src/model.py` y la interfaz CLI acepta $T$ en kelvin y ω en rad/s.
La teoría completa (balance detallado, ergodicidad, soluciones exactas) está en
[docs/fundamentos.md](docs/fundamentos.md), y la revisión contra la rúbrica en
[docs/revision_rubrica.md](docs/revision_rubrica.md).

---

## Instalación

Requiere **Python 3.14** (probado con CPython 3.14.6 en Windows 11).

```bash
python -m venv .venv
```

```bash
# Windows:  .venv\Scripts\activate      Linux/macOS:  source .venv/bin/activate
python -m pip install -r requirements.txt
```

Dependencias con sus versiones ([requirements.txt](requirements.txt)): numpy 2.4.6, scipy 1.17.1,
matplotlib 3.11.0, numba 0.65.1 (llvmlite 0.47.0), pillow 12.2.0 (GIF) y pytest 9.2.0.

## Ejecución

| Qué | Comando | Tiempo aprox. |
|---|---|---|
| Todo: pruebas, datos, figuras, tablas y animación | `python scripts/run_all.py` (o `make all`) | 30–40 min |
| Sólo figuras y tablas desde los datos crudos incluidos | `python scripts/run_all.py --plots-only` (o `make figures`) | ~1 min |
| Verificación rápida de que todo corre | `python scripts/run_all.py --fast` | ~4 min |
| Pruebas automatizadas | `python -m pytest` | ~20 s |
| Reporte y diapositivas (LaTeX) | `make latex`, o `pdflatex` dos veces dentro de `reporte/` y de `presentacion/` | ~1 min |
| Una simulación interactiva (CLI) | `python scripts/interactive_cli.py` | segundos |

`--fast` usa el 5 % de los pasos y escribe en `data/raw_fast/` y `figures/fast/`, así que no
toca los datos del reporte. En Windows `make` no suele estar instalado: `run_all.py` es el
equivalente multiplataforma del `Makefile`.

### Tabla figura → script

Todas las figuras se guardan en `figures/` en PDF (para el reporte) y en PNG (para las
diapositivas). Para regenerar una sola desde los datos crudos:
`python scripts/plot_results.py --only <nombre>`.

| Figura / tabla | Contenido | Datos crudos generados por | Datos en `data/raw/` |
|---|---|---|---|
| `fig_validation` | (a) N = 1 vs $3/(e^\beta-1)$, dos núcleos; (b) N = 3 vs enumeración; (c) discrepancias en σ | `scripts/run_validation.py` | `validation.npz`, `validation_table.csv` |
| `fig_equilibration` | E/N desde dos condiciones iniciales, cruce de cadenas y $n_{burn}$ | `scripts/run_equilibration.py` | `equilibration.npz` |
| `fig_autocorrelation` | C(t), τ_int y método de bloques | `scripts/run_equilibration.py` | `equilibration.npz` |
| `fig_qmax_convergence` | Error de truncamiento vs error estadístico en función de $q_{max}$ | `scripts/run_sweeps.py` (parte A) | `qmax_convergence.npz` |
| `fig_energy_vs_T` | ⟨E⟩/N vs T para BE y FD, N = 10–56, y referencia clásica | `scripts/run_sweeps.py` (parte B) | `sweep/`, `sweep_summary.csv` |
| `fig_condensate_fraction` | $f_0$ vs $T/T_c$ y límite termodinámico | `scripts/run_sweeps.py` (parte B) | `sweep/`, `sweep_summary.csv` |
| `fig_occupations` | ⟨n(ε)⟩: simulación vs canónico exacto vs BE/FD gran canónico | `scripts/run_sweeps.py` (parte B) | `sweep/*_N35_*.npz` |
| `fig_ensemble_comparison` | Canónico − gran canónico y exponente de escalamiento en N | `scripts/run_sweeps.py` (partes B y C) | `sweep_summary.csv`, `ensemble_scaling.npz` |
| `fig_z_scores` | Distribución de discrepancias de todo el barrido vs semi-normal | `scripts/run_sweeps.py` (parte B) | `sweep_summary.csv` |
| `fig_experimental_comparison` | Simulación y límite termodinámico vs experimentos: ⁸⁷Rb (Ensher 1996) y ⁴⁰K (DeMarco y Jin 1999) | `scripts/compare_experiments.py` | `experimental_comparison.npz/.csv` y `data/external/*.csv` |
| `animation_equilibration.gif` | Relajación al equilibrio (presentación) | `scripts/make_animation.py` | `animation.npz` |
| `tabla_validacion.tex` | Tabla de validación para LaTeX | `scripts/run_validation.py` | `validation_table.csv` |
| `tabla_resultados.csv/.tex` | Parámetros calculados por corrida | `scripts/run_sweeps.py` | `sweep_summary.csv` |

Las curvas exactas densas de las figuras se guardan en caché en `data/raw/exact_curves.npz`
y se recalculan solas si faltan.

---

## Estructura

```
├── README.md, requirements.txt, Makefile, pytest.ini, .python-version
├── data/raw/           Datos crudos (.npz/.csv) con metadatos: semilla, parámetros, versiones, commit
├── data/external/      Datos experimentales digitalizados de artículos (con fuente y README)
├── figures/            Figuras finales (PDF + PNG), tablas y animación
├── docs/               fundamentos.md (teoría y justificación), revision_rubrica.md
├── reporte/            Reporte escrito en LaTeX (reporte.tex → reporte.pdf, 5 páginas + apéndices)
├── presentacion/       Diapositivas beamer (presentacion.tex → presentacion.pdf, 14 + 4 de respaldo)
├── src/
│   ├── model.py        Catálogo de estados 3D, Hamiltoniano, unidades SI↔reducidas, inicialización
│   ├── metropolis.py   Núcleos MCMC (Metropolis simétrico y Metropolis–Hastings) con numba
│   ├── analytics.py    Exactos: N = 1, enumeración, función generatriz canónica, gran canónico, T_c, ε_F
│   ├── stats.py        τ_int (Sokal / primer cero), método de bloques, discrepancia z, equilibración
│   ├── observables.py  E/N, f_0 = ⟨n_0⟩/N, ⟨n(ε)⟩ con incertidumbre y comparación con referencias
│   ├── datastore.py    Guardado/lectura de datos crudos con metadatos; tablas CSV
│   ├── plotting.py     Todas las figuras del reporte
│   └── config.py       Parámetros globales: N, malla de T, pasos, semillas (BASE_SEED)
├── scripts/
│   ├── run_validation.py    Validación cuantitativa (A2)
│   ├── run_equilibration.py Termalización desde 2 estados iniciales y autocorrelación (A3)
│   ├── run_sweeps.py        Barridos en q_max, T y N; escalamiento canónico vs gran canónico
│   ├── plot_results.py      Lee data/raw/ y regenera todas las figuras y tablas
│   ├── make_animation.py    GIF de la relajación al equilibrio
│   ├── compare_experiments.py Comparación con datos experimentales (equilibrio, variables escaladas)
│   ├── digitize_external.py Digitalización de figuras publicadas (requiere Poppler y los PDF)
│   ├── interactive_cli.py   Interfaz de usuario en línea de comandos
│   └── run_all.py           Función integradora (equivalente a `make all`)
└── tests/
    ├── test_validation.py   Prueba automatizada del caso de validación (N = 1 y N = 2, seed = 42)
    └── test_core.py         Catálogo, exactos vs enumeración, conservación de N, Pauli,
                             distribución estacionaria por microestado, τ_int de un AR(1)
```

### Módulos (resumen de funciones)

- **`model`**: `build_state_catalog(q_max)` lista los $M$ estados con $\varepsilon\le q_{max}$
  (cada estado degenerado por separado, ordenados por energía). `initialize_configuration(N, M,
  statistics, mode, rng)` crea un estado base (`"ground"`) o una configuración aleatoria a $T=\infty$
  (`"excited"`). `compute_total_energy(n, energies, omega=None)` evalúa $H(\mathbf n)$, en J si se
  da ω. `reduced_temperature`, `kelvin_temperature` y `energy_to_joules` convierten unidades.
- **`metropolis`**: `metropolis_step` (núcleo simétrico de la propuesta aprobada),
  `metropolis_hastings_step` (partícula al azar con factor de Hastings) y
  `run_mcmc_chain(N, beta, q_max, statistics, n_steps, n_burn, sample_interval, seed, init_mode, proposal)`,
  que devuelve las trazas de E y de las ocupaciones por nivel, la aceptación y los metadatos.
- **`analytics`**: `exact_single_particle_energy`, `exact_canonical_enumeration` (fuerza bruta),
  `exact_canonical` (cualquier N, funciones generatrices), `grand_canonical_observables` (μ fijado por N),
  `choose_q_max` (corte con truncamiento < 10⁻⁴T), `bec_critical_temperature` y `fermi_energy`.
- **`stats`**: `integrated_autocorrelation_time`, `statistical_uncertainty`
  ($\sigma_{\bar A}=\sqrt{2\tau_{int}\sigma_A^2/N}$), `blocking_error`, `blocking_analysis`,
  `discrepancy_in_sigmas` y `estimate_burn_in`.
- **`observables`**: `compute_observables(E_trace, level_trace, N)`, `compare_with_reference`.

## Reproducibilidad y semillas

- Todas las semillas salen de `BASE_SEED = 20261112` (`src/config.py`) con
  `make_seed(*claves)` (`numpy.random.SeedSequence`). La semilla entera de cada corrida
  queda en su `.npz` (campo `metadata`), en `sweep_summary.csv` y en `validation_table.csv`.
- Generador: `numpy.random.default_rng(seed)` (PCG64), usado dentro de los núcleos de numba.
  Misma semilla ⇒ misma cadena, bit a bit (`test_seed_reproducibility`).
- Cada `.npz` guarda la fecha, el commit de git y las versiones de Python, numpy, scipy,
  numba y matplotlib.
- La prueba del caso de validación usa `seed = 42`.

## Interfaz de usuario (CLI)

```bash
python scripts/interactive_cli.py
```

Pregunta la estadística (bosones o fermiones), N, la temperatura (reducida, o en kelvin junto
con ω en rad/s), el corte, el núcleo, los pasos y la semilla (si se deja vacía se genera una y
se muestra). Al terminar imprime E/N y $f_0$ con σ, la discrepancia en σ y el **error
porcentual** frente a la solución canónica exacta, la desviación respecto a BE/FD gran
canónico, μ, τ_int, la aceptación y la **semilla usada**. Opcionalmente guarda los datos
crudos y una figura de ⟨n(ε)⟩. Modo directo, sin preguntas:

```bash
python scripts/interactive_cli.py --statistics fermions --N 20 --T 1.5 --seed 42 --plot
```

## Resultados principales

Corrida completa con las semillas de `src/config.py`. Las incertidumbres son 1σ, corregidas por autocorrelación.

**Validación (A2).**
- 68 comparaciones contra resultados exactos: N = 1 contra la fórmula analítica, N = 3 contra
  enumeración y los dos núcleos con N = 10. Discrepancia media z = 0.77, máxima 2.48, y 97 % con z < 2.
- La función generatriz y la enumeración coinciden hasta 1.2 × 10⁻¹³.
- En las 104 corridas del barrido (BE y FD, N = 10–56, T = 0.25–5) cada punto se compara con la
  solución canónica exacta: z(E/N) tiene media 0.83 (la distribución semi-normal ideal da
  0.80) y 95 % de los puntos con z < 2; para z(f₀), 99 % con z < 2.
- Como comprobación adicional, se combinaron 12 semillas independientes en tres puntos de
  control. Los promedios coinciden con el exacto (z ≤ 1.1) y la dispersión entre semillas
  coincide con σ (cociente 0.86–1.23): los errores están bien calibrados.

**Equilibración e incertidumbre (A3).**
- Las cadenas fría y caliente se cruzan a 6.8 × 10⁵ pasos (bosones, N = 35, T = 2) y a
  3.6 × 10⁴ pasos (fermiones, N = 35, T = 1). Es muy por debajo de la termalización descartada
  (2.4 × 10⁷ y 6 × 10⁶). En producción las dos cadenas coinciden (z = 0.47 y 0.18).
- Para bosones, τ_int ≈ 12 mediciones: el error corregido es 5 veces el ingenuo, y el método de bloques lo confirma.

**Condensación de Bose–Einstein (tamaño finito).**
- A T/T_c ≈ 0.49 se obtiene f₀ = 0.627 ± 0.002, 0.674 ± 0.003, 0.712 ± 0.003 y 0.735 ± 0.004
  para N = 10, 20, 35 y 56. Es muy por debajo del límite termodinámico 1 − (T/T_c)³ ≈ 0.88,
  pero se acerca a él al crecer N y sigue la corrección de tamaño finito
  1 − t³ − 3C t² N^{-1/3} (0.633, 0.690, 0.726, 0.750).
- Con N finito el condensado aparece como un cruce gradual, no como una transición abrupta (`fig_condensate_fraction`).

**Mar de Fermi.**
- A T = 0.25: E/N = 1.5729 ± 0.0005, 2.3161 ± 0.0003, 3.0581 ± 0.0002 y 3.8010 ± 0.0002, frente
  a E₀/N = 1.5, 2.25, 3.0 y 3.75 del mar lleno (capas cerradas).
- La ocupación forma un escalón en ε_F que se suaviza al subir T (`fig_occupations`). A T alta,
  bosones y fermiones convergen a la referencia clásica (`fig_energy_vs_T`).

**¿Cumple con Bose–Einstein / Fermi–Dirac? (comparación entre ensembles).**
- La simulación reproduce la forma de BE y FD (`fig_occupations`), pero con N fijo hay
  desviaciones medibles respecto al ensemble gran canónico:
  - el condensado canónico es hasta 0.074 menor (N = 10, T/T_c ≈ 0.6);
  - la energía de los bosones es hasta 16 % mayor;
  - la de los fermiones es hasta 1.6 % menor.
- Esas diferencias decaen aproximadamente como 1/N: N^{-0.86 ± 0.06} en bosones a T/T_c = 0.5
  (el exacto con el mismo ajuste da −0.93) y N^{-1.13 ± 0.07} en fermiones a T/T_F = 0.25
  (exacto −1.11). Las distribuciones FD y BE se recuperan en el límite N → ∞ (`fig_ensemble_comparison`).

**Comparación con experimentos** (`fig_experimental_comparison`; datos digitalizados en
[data/external/](data/external/README.md)). Sólo se comparan propiedades de equilibrio, en
variables escaladas. Los pasos de Monte Carlo no son tiempo físico, así que la "evolución" no es comparable.
- *Fermiones*, ⁴⁰K, N ≈ 7×10⁵ (DeMarco y Jin 1999):
  - El exceso de energía δU/U_cl sigue la forma del gas de Fermi ideal.
  - Con la escala nominal de T/T_F el ajuste es malo: χ² = 201 con 29 grados de libertad.
  - Si se reescala T/T_F por s = 0.847 ± 0.009, el acuerdo es excelente (χ² = 30.2/28, p = 0.36).
    Ese 15 % cabe en el 20 % de incertidumbre sistemática en T/T_F que declaran los autores.
  - La simulación converge al mismo límite al crecer N: la desviación media baja de
    +0.0086 (N = 10) a +0.0039 (N = 56).
- *Bosones*, ⁸⁷Rb (Ensher et al. 1996):
  - Por encima de T₀ la energía medida está mucho más cerca del gas de Bose ideal
    (−0.060 ± 0.020 k_BT₀) que del clásico (−0.157). Esa diferencia equivale a ~4 % en la escala T/T₀,
    comparable con la incertidumbre de T_c = 0.94(5) T₀.
  - Por debajo de 0.9 T₀ el dato supera al ideal en +0.31 k_BT₀: es la energía de interacción
    del condensado, que el modelo ideal no incluye. Es la principal limitación física del modelo.
  - Con N pequeño la simulación también queda por encima del ideal a T baja, pero por otra
    razón: la energía de punto cero es (3/4)ħω/k_BT₀ ∝ N^{-1/3}.

## Uso de herramientas de IA

Parte del código y de la documentación se generó con ayuda de un asistente de IA (Claude).
Todo el código se verificó con las pruebas automatizadas, con la comparación contra la
solución exacta y con la revisión de las derivaciones en `docs/fundamentos.md`. El reporte
debe incluir la declaración correspondiente (rúbrica §8).
