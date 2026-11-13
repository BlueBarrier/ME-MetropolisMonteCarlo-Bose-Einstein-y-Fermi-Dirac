# Estadística cuántica por Monte Carlo: condensación de Bose–Einstein vs. mar de Fermi

Proyecto final de **Mecánica Estadística** (Universidad del Valle de Guatemala).

Simulación de gases ideales de **bosones** y **fermiones** (una componente de espín) en una trampa armónica isotrópica 3D, en el **ensemble canónico** ($N$, $T$, $\omega$ fijos), con **Algoritmo de Metrópolis de Monte Carlo** sobre configuraciones de ocupación. Se valida contra resultados exactos y se estudia cómo la antisimetría de la función de onda produce el llenado del mar de Fermi y la simetría de la función de onda produce la acumulación de los estados en la energía base para el condensado de Bose-Einstein, en función de $T$ y $N$.

También se mide el calor específico (transición lambda de los bosones y ley lineal de los fermiones) y el potencial químico de los fermiones.

---

**Integrantes:** Isabella Ramírez, Erick Barrera, José Dávila.

---

## Modelo Físico

$$
H(\mathbf n)=\sum_i n_i\,\varepsilon_i \quad \text{Hamiltoniano dada la energía y el núm. de partículas.}\\ 

\varepsilon_i=\hbar\omega(q_x+q_y+q_z)\quad \text{Energía por nivel dado un número cuántico por eje.}\\\\ 
n = q_x + q_y + q_z \quad \text{Número cuántico principal}\\\\

\sum_i n_i=N \quad \text{Ligadura para ambas estadísticas (no se consideran fotones).}\\

n_i\in\{0,\dots,N\}\ \text{(BE)},\ \{0,1\}\ \text{(FD)} \quad \text{Número de  ocupación por nivel.}\\\\ 

P(\mathbf n)\propto e^{-\beta H(\mathbf n)} \quad \text{Probabilidad dado n}
$$

Con unidades reducidas: $\hslash \omega=k_b= 1$. Las conversiones al SI están en la interfaz CLI y acepta $T$ en kelvin y ω en rad/s. 

---

## Instalación

Requiere **Python 3.14** (probado con CPython 3.14.6 en Windows 11).

Recomendamos crear un *virtual enviroment* para no tener conflictos con otras librearías en contar ya con python 3.14.6

```bash
python -m venv .venv
```

Para instalar las librerías necesarias solo se debe correr el archivo *requirements.txt*:

```bash
# Windows:  .venv\Scripts\activate      Linux/macOS:  source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Ejecución

Todo se maneja desde un menú:

```bash
python scripts/run.py
```

| Opción | Qué hace | Script | Tiempo aprox. |
|---|---|---|---|
| 1 | Simulación individual (bosones o fermiones) con resumen de errores | `scripts/interactive_cli.py` | segundos |
| 2 | Comparar datos teóricos propios con la simulación, con el q_max que elija el usuario | `scripts/compare_theory.py` | ~10 s por punto |
| 3 | Validación contra resultados exactos | `scripts/run_validation.py` | ~3 min |
| 4 | Equilibración, autocorrelación y verificación de n_burn en todo el barrido | `scripts/run_equilibration.py` | ~5 min |
| 5 | Barridos en q_max, T y N, y ocupaciones con N = 45 | `scripts/run_sweeps.py` | ~20 min |
| 6 | Generar las figuras (PNG en `fig/`) y verlas una por una | `scripts/plot_results.py` | ~30 s |
| 7 | Ver las figuras ya guardadas | `scripts/view_figures.py` | — |
| 8 | Animación de la relajación al equilibrio (GIF, N = 56) | `scripts/make_animation.py` | ~1 min |
| 9 | Pruebas automatizadas | `python -m pytest` | ~15 s |
| 10 | Reproducir todo (pruebas, 3, 4, 5, 8 y figuras) | — | 30–40 min |
| r | Activar o desactivar el modo rápido (5 % de los pasos, en `data/raw_fast/` y `fig/fast/`) | — | — |

Cada script también se puede correr por separado (por ejemplo `python scripts/plot_results.py --no-show`).

### Comparar datos teóricos propios con la simulación

`scripts/compare_theory.py` (opción 2 del menú) pide la estadística, N, el corte **q_max** y un
observable, y los valores teóricos como pares (x, y), desde un CSV de dos columnas o escritos a mano:

| Observable | x | y | Simulaciones |
|---|---|---|---|
| `energy` | T (ħω/k_B) | ⟨E⟩/N (ħω) | una por cada T |
| `condensate` | T | f₀ = ⟨n₀⟩/N | una por cada T |
| `occupation` | nivel ε (entero ≤ q_max) | ⟨n(ε)⟩ por estado | una, a la T que se indique |

Corre la simulación con ese q_max e imprime, para cada punto, la teoría, la simulación ± σ, la
discrepancia z = (sim − teo)/σ, el error porcentual y el error de truncamiento estimado por usar ese
q_max. Guarda la tabla en `data/raw/comparaciones/` y la figura (teoría frente a simulación, con
residuos z) en `fig/comparaciones/`, y la abre en la ventana. Hay dos ejemplos en `data/teoria/`:

```bash
python scripts/compare_theory.py --statistics fermions --N 20 --q-max 30 --observable energy --theory data/teoria/ejemplo_fermiones_N20_energia.csv
```

```bash
python scripts/compare_theory.py --statistics bosons --N 20 --q-max 15 --observable occupation --T 1 --theory data/teoria/ejemplo_bosones_N20_ocupacion_T1.csv
```

### Tabla figura → script

Todas las figuras se guardan en `fig/` en PNG, numeradas en orden de lectura. Al terminar,
`plot_results.py` abre una ventana para verlas una por una (← / → o los botones; cierre la ventana
para salir); `python scripts/view_figures.py` la abre de nuevo sin regenerarlas.

Para regenerar una sola desde los datos crudos:
`python scripts/plot_results.py --only <nombre interno>` (por ejemplo `fig_validation`; añada
`--no-show` para no abrir la ventana).

| Archivo | Nombre interno (`--only`) | Contenido | Datos crudos generados por | Datos en `data/raw/` |
|---|---|---|---|---|
| `fig/01a_validacion_N1_energia_vs_formula_analitica.png` | `fig_validation` | N = 1 frente a $3/(e^eta-1)$ | `scripts/run_validation.py` | `validation.npz` |
| `fig/01b_validacion_N3_ocupaciones_vs_enumeracion.png` | `fig_validation` | N = 3: histograma de ⟨n(ε)⟩ frente a la enumeración (BE y FD) | `scripts/run_validation.py` | `validation.npz` |
| `fig/02a_equilibracion_bosones_desde_dos_estados_iniciales.png` | `fig_equilibration` | Bosones: E/N desde dos condiciones iniciales, cruce de cadenas y $n_{burn}$ | `scripts/run_equilibration.py` | `equilibration.npz` |
| `fig/02b_equilibracion_fermiones_desde_dos_estados_iniciales.png` | `fig_equilibration` | Fermiones: E/N desde dos condiciones iniciales, cruce de cadenas y $n_{burn}$ | `scripts/run_equilibration.py` | `equilibration.npz` |
| `fig/03_autocorrelacion_de_la_energia.png` | `fig_autocorrelation` | C(t) y τ_int; cuánto subestima el error ingenuo | `scripts/run_equilibration.py` | `equilibration.npz` |
| `fig/04_convergencia_en_el_corte_energetico_qmax.png` | `fig_qmax_convergence` | Error de truncamiento vs error estadístico en función de $q_{max}$ | `scripts/run_sweeps.py` (parte A) | `qmax_convergence.npz` |
| `fig/05_energia_por_particula_vs_temperatura.png` | `fig_energy_vs_T` | ⟨E⟩/N vs T para BE y FD, N = 10–56, y referencia clásica | `scripts/run_sweeps.py` (parte B) | `sweep/`, `sweep_summary.csv` |
| `fig/06_fraccion_de_condensado_bosones_vs_T.png` | `fig_condensate_fraction` | $f_0$ vs $T/T_c$ y límite termodinámico | `scripts/run_sweeps.py` (parte B) | `sweep/`, `sweep_summary.csv` |
| `fig/07_ocupaciones_bosones_vs_fermiones_N45.png` | `fig_occupations` | Histograma del número medio de ocupación ⟨n(ε)⟩ con N = 45: simulación (barras) vs canónico exacto | `scripts/run_sweeps.py` (parte C) | `occupations.npz` |
| `fig/08a_calor_especifico_bosones_transicion_lambda.png` | `fig_heat_capacity_bosons` | $C_V/Nk_B$ de bosones vs $T/T_c$ (fluctuaciones de E), canónico exacto y teoría $N\to\infty$: transición lambda | `scripts/run_sweeps.py` (parte B) | `sweep_summary.csv` |
| `fig/08b_calor_especifico_fermiones.png` | `fig_heat_capacity_fermions` | $C_V/Nk_B$ de fermiones vs $T/E_F$: ley lineal de Sommerfeld y límite clásico 3 | `scripts/run_sweeps.py` (parte B) | `sweep_summary.csv` |
| `fig/08c_potencial_quimico_fermiones.png` | `fig_chemical_potential_fermions` | μ(T) de fermiones: ajuste de Fermi–Dirac a ⟨n(ε)⟩ simulado vs teoría | `scripts/run_sweeps.py` (parte B) | `sweep_summary.csv` |
| `fig/09_animacion_relajacion_al_equilibrio_N56.gif` | — | Relajación al equilibrio (N = 56): ~8 s de relajación hasta n_burn y los últimos ~4 s en equilibrio, con el promedio acumulado sobre el exacto | `scripts/make_animation.py` | `animation.npz` |
| `fig/comparaciones/*.png` | — | Teoría del usuario vs simulación | `scripts/compare_theory.py` | `comparaciones/*.csv` |
| `data/tablas/tabla_validacion.tex` | — | Tabla de validación (LaTeX) | `scripts/run_validation.py` | `validation_table.csv` |
| `data/tablas/tabla_resultados.csv/.tex` | — | Parámetros calculados por corrida | `scripts/run_sweeps.py` | `sweep_summary.csv` |

Las curvas exactas densas de las figuras se guardan en caché en `data/raw/exact_curves.npz`
y se recalculan solas si faltan.

---

## Estructura

```
├── README.md, requirements.txt, pytest.ini, .python-version
├── data/raw/           Datos crudos (.npz/.csv) con metadatos: semilla, parámetros, versiones, commit
├── data/teoria/        Ejemplos de datos teóricos para comparar (CSV de dos columnas)
├── data/tablas/        Tablas de resultados (CSV y LaTeX)
├── fig/                Figuras PNG numeradas (01_ … 08_), animación GIF (09_) y comparaciones/
├── docs/               fundamentos.md (teoría y justificación), revision_rubrica.md
├── src/
│   ├── model.py          Catálogo de estados 3D, Hamiltoniano, unidades SI↔reducidas, inicialización
│   ├── metropolis.py     Algoritmo de Metropolis–Hastings (MCMC) con numba
│   ├── analytics.py      Exactos: N = 1, enumeración, función generatriz canónica, gran canónico, T_c, ε_F
│   ├── stats.py          τ_int (Sokal / primer cero), error corregido, discrepancia z, equilibración
│   ├── observables.py    E/N, f_0 = ⟨n_0⟩/N, ⟨n(ε)⟩ con incertidumbre y comparación con referencias
│   ├── theory_compare.py Comparación de la simulación con datos teóricos del usuario
│   ├── datastore.py      Guardado/lectura de datos crudos con metadatos; tablas CSV
│   ├── plotting.py       Todas las figuras (PNG en fig/ con nombres descriptivos)
│   ├── viewer.py         Ventana para ver las figuras una por una (anterior / siguiente)
│   └── config.py         Parámetros globales: N, malla de T, pasos, semillas (BASE_SEED)
├── scripts/
│   ├── run.py               Menú principal: elige qué ejecutar
│   ├── interactive_cli.py   Simulación individual en línea de comandos
│   ├── compare_theory.py    Teoría del usuario vs simulación con q_max a elección
│   ├── run_validation.py    Validación contra resultados exactos
│   ├── run_equilibration.py Termalización desde 2 estados iniciales y autocorrelación
│   ├── run_sweeps.py        Barridos en q_max, T y N; ocupaciones con N = 45
│   ├── plot_results.py      Lee data/raw/, regenera figuras (fig/) y tablas (data/tablas/) y abre el visor
│   ├── view_figures.py      Abre la ventana con las figuras de fig/ sin regenerarlas
│   └── make_animation.py    GIF de la relajación al equilibrio (N = 56)
└── tests/
    ├── test_validation.py   Prueba automatizada del caso de validación (N = 1 y N = 2, seed = 42)
    └── test_core.py         Catálogo, exactos vs enumeración, conservación de N, Pauli,
                             distribución estacionaria por microestado, τ_int de un AR(1),
                             comparación con teoría del usuario
```

### Módulos (resumen de funciones)

- **`model`**: `build_state_catalog(q_max)` lista los $M$ estados con $\varepsilon\le q_{max}$
  (cada estado degenerado por separado, ordenados por energía). `initialize_configuration(N, M,
  statistics, mode, rng)` crea un estado base (`"ground"`) o una configuración aleatoria a $T=\infty$
  (`"excited"`). `compute_total_energy(n, energies, omega=None)` evalúa $H(\mathbf n)$, en J si se
  da ω. `reduced_temperature`, `kelvin_temperature` y `energy_to_joules` convierten unidades.
- **`metropolis`**: `metropolis_hastings_step` (partícula al azar con factor de Hastings) y
  `run_mcmc_chain(N, beta, q_max, statistics, n_steps, n_burn, sample_interval, seed, init_mode)`,
  que devuelve las trazas de E y de las ocupaciones por nivel, la aceptación y los metadatos.
- **`analytics`**: `exact_single_particle_energy`, `exact_canonical_enumeration` (fuerza bruta),
  `exact_canonical` (cualquier N, funciones generatrices), `grand_canonical_observables` (μ fijado por N),
  `choose_q_max` (corte con truncamiento < 10⁻⁴T), `bec_critical_temperature` y `fermi_energy`.
- **`stats`**: `integrated_autocorrelation_time`, `statistical_uncertainty`
  ($\sigma_{\bar A}=\sqrt{2\tau_{int}\sigma_A^2/N}$), `discrepancy_in_sigmas` y `estimate_burn_in`.
- **`observables`**: `compute_observables(E_trace, level_trace, N)`, `compare_with_reference`.
- **`theory_compare`**: `compare_with_theory(statistics, N, q_max, observable, theory_x, theory_y, T, ...)`,
  `read_theory_file(path)` y `parse_pairs(texto)`.

## Reproducibilidad y semillas

- Todas las semillas salen de `BASE_SEED = 20261112` (`src/config.py`) con
  `make_seed(*claves)` (`numpy.random.SeedSequence`). La semilla entera de cada corrida
  queda en su `.npz` (campo `metadata`), en `sweep_summary.csv` y en `validation_table.csv`.
- Generador: `numpy.random.default_rng(seed)` (PCG64), usado dentro de las funciones compiladas con numba.
  Misma semilla ⇒ misma cadena, bit a bit (`test_seed_reproducibility`).
- Cada `.npz` guarda la fecha, el commit de git y las versiones de Python, numpy, scipy,
  numba y matplotlib.
- La prueba del caso de validación usa `seed = 42`.

## Interfaz de usuario (CLI)

```bash
python scripts/interactive_cli.py
```

Pregunta la estadística (bosones o fermiones), N, la temperatura (reducida, o en kelvin junto
con ω en rad/s), el corte, los pasos y la semilla (si se deja vacía se genera una y
se muestra). Al terminar imprime E/N y $f_0$ con σ, la discrepancia en σ y el **error
porcentual** frente a la solución canónica exacta, la desviación respecto a BE/FD gran
canónico, μ, τ_int, la aceptación y la **semilla usada**. Opcionalmente guarda los datos
crudos y una figura de ⟨n(ε)⟩. Modo directo, sin preguntas:

```bash
python scripts/interactive_cli.py --statistics fermions --N 20 --T 1.5 --seed 42 --plot
```

## Resultados principales

Corrida completa con las semillas de `src/config.py`. Las incertidumbres son 1σ, corregidas por autocorrelación.

**Validación.**
- 29 comparaciones contra resultados exactos, todas con Metropolis–Hastings: N = 1 contra la fórmula
  analítica, N = 3 contra enumeración y N = 10 contra la solución canónica exacta. Discrepancia media
  z = 0.73, máxima 1.92, y 100 % con z < 2.
- En las 152 corridas del barrido (BE y FD, N = 10–56, 19 temperaturas entre 0.25 y 10) cada punto se
  compara con la solución canónica exacta: z(E/N) tiene media 0.90 y 95 % de los puntos con z < 2;
  z(f₀), 97 %; z(C_V), 93 %.

**Equilibración e incertidumbre.**
- Las cadenas fría y caliente se cruzan a 1.5 × 10⁶ pasos (bosones, N = 35, T = 2) y a 3.6 × 10⁴
  pasos (fermiones, N = 35, T = 1), muy por debajo de la termalización descartada (9.6 × 10⁷ y
  6 × 10⁶). En producción las dos cadenas coinciden (z = 1.71 y 0.18).
- τ_int ≈ 3.2 mediciones en bosones y 0.6 en fermiones: el error corregido es 2.5 y 1.1 veces el ingenuo.
- **La termalización es suficiente en todo el barrido**: n_burn = 10 % de los pasos (`BURN_FRACTION`),
  y `run_equilibration.py` comprueba los 152 puntos: todos cruzan antes de n_burn; el peor caso
  (bosones, N = 56, T = 10) al 41 %. Resultado en `data/raw/burn_check.csv`.

**Condensación de Bose–Einstein (tamaño finito).**
- A T/T_c ≈ 0.49: f₀ = 0.627 ± 0.002, 0.674 ± 0.003, 0.710 ± 0.002 y 0.735 ± 0.002 para N = 10, 20,
  35 y 56, por debajo del límite termodinámico 1 − (T/T_c)³ ≈ 0.885 (`06_fraccion_de_condensado_bosones_vs_T.png`).
- El calor específico tiene un máximo redondeado en T/T_c ≈ 0.75 que crece con N: 3.73 ± 0.04 (N = 10)
  a 5.14 ± 0.09 (N = 56), frente al pico singular 12ζ(4)/ζ(3) ≈ 10.8 del límite N → ∞
  (`08a_calor_especifico_bosones_transicion_lambda.png`).

**Mar de Fermi.**
- A T = 0.25: E/N = 1.5729 ± 0.0005 (N = 10) frente a E₀/N = 1.5 del mar lleno.
- C_V es lineal en T a baja temperatura (π² T/T_F) y tiende a 3 k_B; μ cambia de signo en T/T_F ≈ 0.57
  y el μ ajustado coincide con Fermi–Dirac (discrepancia media 0.6σ) (`08b_…`, `08c_…`).

## Uso de herramientas de IA

Parte del código y de la documentación se generó con ayuda de un asistente de IA (Claude).
Todo el código se verificó con las pruebas automatizadas, con la comparación contra la
solución exacta..
