"""Gas ideal de bosones y fermiones en una trampa armónica 3D (ensemble canónico).

Paquete del proyecto final de Mecánica Estadística (UVG). Módulos:

- ``model``       : catálogo de estados, Hamiltoniano, unidades e inicialización.
- ``metropolis``  : núcleos de Markov Chain Monte Carlo (Metropolis y Metropolis–Hastings).
- ``analytics``   : soluciones exactas (N = 1, enumeración, función generatriz canónica,
                    ensemble gran canónico y límite termodinámico).
- ``stats``       : equilibración, autocorrelación integrada y discrepancia z.
- ``observables`` : energía media, ocupaciones por nivel y fracción de condensado.
- ``datastore``   : guardado/lectura de datos crudos con metadatos (semilla, versiones).
- ``plotting``    : figuras PNG (fig/) a partir de los datos crudos.
- ``viewer``      : ventana para ver las figuras una por una.
- ``theory_compare``: comparación de la simulación con datos teóricos del usuario.
"""

__version__ = "1.0.2"
