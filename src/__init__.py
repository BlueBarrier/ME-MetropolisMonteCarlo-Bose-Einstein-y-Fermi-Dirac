"""Gas ideal de bosones y fermiones en una trampa armónica 3D (ensemble canónico).

Paquete del proyecto final de Mecánica Estadística (UVG). Módulos:

- ``model``       : catálogo de estados, Hamiltoniano, unidades e inicialización.
- ``metropolis``  : núcleos de Markov Chain Monte Carlo (Metropolis y Metropolis–Hastings).
- ``analytics``   : soluciones exactas (N = 1, enumeración, función generatriz canónica,
                    ensemble gran canónico y límite termodinámico).
- ``stats``       : equilibración, autocorrelación integrada, método de bloques.
- ``observables`` : energía media, ocupaciones por nivel y fracción de condensado.
- ``datastore``   : guardado/lectura de datos crudos con metadatos (semilla, versiones).
- ``plotting``    : figuras del reporte a partir de los datos crudos.
"""

__version__ = "1.0.0"
