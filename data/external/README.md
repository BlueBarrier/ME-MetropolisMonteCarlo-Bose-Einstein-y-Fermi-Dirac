# Datos experimentales digitalizados

Puntos experimentales extraídos de figuras publicadas, para comparar en equilibrio con la
simulación y con el límite termodinámico (`scripts/compare_experiments.py`,
figura `fig_experimental_comparison`). Los PDF originales tienen derechos de autor y **no se
incluyen**: sólo se guardan los puntos digitalizados, con su fuente.

| Archivo | Fuente | Qué contiene |
|---|---|---|
| `demarco_jin_1999_fig4.csv` | B. DeMarco y D. S. Jin, *Onset of Fermi degeneracy in a trapped atomic gas*, Science **285**, 1703 (1999), Fig. 4. | 29 puntos de $\delta U/U_{cl}=(U-3Nk_BT)/3Nk_BT$ vs $T/T_F$ para ⁴⁰K (N ≈ 7×10⁵), con $k_BT_F=\hbar(6\omega_z\omega_r^2N)^{1/3}$. Incertidumbre estadística típica: la barra de error de la figura, medida como σ = 0.0072. Los autores declaran además un 20 % de incertidumbre sistemática en $T/T_F$. |
| `ensher_1996_energy_via_cornell_review_fig5.csv` | J. R. Ensher, D. S. Jin, M. R. Matthews, C. E. Wieman y E. A. Cornell, PRL **77**, 4984 (1996), en la figura reproducida en E. A. Cornell, J. R. Ensher y C. E. Wieman, *Experiments in dilute atomic Bose-Einstein condensation*, arXiv:cond-mat/9903109, Fig. 5. | 38 puntos de energía de liberación (cinética + interacción) por partícula $E_{rel}/(Nk_BT_0)$ vs $T/T_0$ para ⁸⁷Rb, con $k_BT_0=\hbar\bar\omega(N/\zeta(3))^{1/3}$. La figura no tiene barras de error; la dispersión se estima de los propios datos. $T_c$ medido: 0.94(5) $T_0$. |
| `check_*.png` (no se versiona) | — | Imágenes de control que genera `digitize_external.py`: cada punto detectado está marcado en rojo sobre la figura original y las marcas usadas para calibrar los ejes en azul. Están en `.gitignore` porque contienen la figura original, que tiene derechos de autor. |

## Cómo se digitalizó

Con `scripts/digitize_external.py` (requiere `pdftoppm` de Poppler):

1. La página se renderiza a 600 dpi.
2. Los ejes se calibran con las marcas principales (ajuste lineal, residuo < 10⁻³ en unidades de los datos).
3. Los marcadores rellenos se detectan con apertura morfológica, que elimina líneas y texto; las manchas solapadas se separan con k-medias.
4. La barra de error típica se mide recorriendo el trazo vertical que atraviesa su marcador.

La resolución es de ~3×10⁻³ (x) y 2×10⁻⁴ (y) por píxel en DeMarco–Jin, y de ~7×10⁻⁴ y
1×10⁻³ en Ensher. En ambos casos es despreciable frente a las incertidumbres experimentales.

Para repetirlo, consiga los PDF (el de arXiv es de acceso abierto:
https://arxiv.org/abs/cond-mat/9903109; el de Science está en la revista o en el repositorio de
NIST/JILA) y ejecute:

```bash
python scripts/digitize_external.py --demarco RUTA/DeMarco_Jin_1999.pdf --cornell RUTA/cond-mat_9903109.pdf
```
