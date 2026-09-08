# Spectral-reduction validation

Python validation of the mathematics behind a **non-spectral, real-time model of
fluorescent materials**, and of its port to a Godot 4.6 shader.  The model is
the one of Belcour, Fichet & Barla (SIGGRAPH 2025), built on the non-orthogonal
spectral reduction of Fichet, Belcour & Barla (2024).

The code was checked against a full per-wavelength reference before being
ported to shader code.  Clone the repository, create the environment, and every
figure and check regenerates from the five scripts in `scripts/`.

`doc/main.tex` covers the theory, the experiments and the conventions of the
Godot port.  Usage is described below.

## Layout

```
scripts/
  validate_reflectance.py      reflectance only, 3-channel XYZ  -> results/reflectance/
  validate_fluorescence.py     full fluorescence, 4-channel XYZU -> results/fluorescence/
  fig_upscale_diag.py          diagonal vs reduced transport     -> results/upscale_diag/
  generate_godot_matrices.py   prints the constants the Godot addon hardcodes
  verify_fbar_convention.py    check: controller F-bar == validated F-bar
  verify_godot_shader.py       check: shader algebra == validated single bounce
  _paths.py                    repository-relative paths (run scripts from anywhere)
data/
  cmf/xyzu.csv                 XYZU colour-matching functions, 300-799 nm (Fichet 2024)
  cmf/ciexyz06_2deg.csv        CIE 2006 2-degree XYZ (Belcour 2025 supplement)
  srgb.coeff                   Jakob & Hanika sRGB spectral-upsampling table
results/                       committed reference figures (regenerable)
doc/                           LaTeX companion document (Overleaf-ready)
```

## Quick start

```sh
make venv            # python3 -m venv .venv + pinned requirements
make check           # ~1 s: the two Godot-port checks, exit 1 on any failure
make reflectance     # ~1 min
make fluorescence    # several minutes (both U-channel methods + sweeps)
make upscale         # ~10 s
make all             # everything above
```

Without `make`: `.venv/bin/python scripts/<name>.py` from any directory.

Requires Python 3.12 and the pinned versions in `requirements.txt`
(numpy, scipy, matplotlib, colour-science).

## What each script establishes

| Script | Question it answers | Output |
|---|---|---|
| `validate_reflectance.py` | Does the reduced reflectance matrix track a spectral reference over 20 bounces better than a per-channel product, and is the *Hybrid* scheme (diagonal first bounce, reduced afterwards) sound? Also fixes the albedo alpha-scaling. | `results/reflectance/<material>.png`, 8 materials x 6 illuminants |
| `validate_fluorescence.py` | Does the closed-form Gaussian fluorescence model (the thing a shader evaluates) reproduce a full spectral reference in XYZU, and how do the two U-channel albedo methods compare? | `results/fluorescence/method{1,2}/`, plus sweeps and UV tests |
| `fig_upscale_diag.py` | Reproduction of the authors' supplementary multi-bounce figure (diagonal vs reduced) with plain numpy/matplotlib. | `results/upscale_diag/` |
| `generate_godot_matrices.py` | Prints the fixed matrices the Godot addon hardcodes, with self-checks, so the engine uses the same basis as the Python scripts. | stdout (GLSL + GDScript literals) |
| `verify_fbar_convention.py` | Checks that the Godot controller's F-bar equals the validated one to machine precision, and that the earlier index convention did not. | pass/fail, exit code |
| `verify_godot_shader.py` | Checks that the shader's `R c + F-bar (I - R) c` reproduces the validated single-bounce output for both reduced and diagonal reflectance. | pass/fail, exit code |

Current check values (from `make check`):

| Check | Value |
|---|---|
| corrected controller F-bar vs validated F-bar | 8.0e-17 |
| earlier controller effective operator vs validated F-bar | 5.3e-01 (the bug that was fixed) |
| shader output vs validated, reduced reflectance | 1.2e-16 |
| shader output vs validated, diagonal reflectance | 1.1e-16 |
| analytic vs brute-force F-bar (approximation error of the Gaussian model) | 1.7e-01 |

## Reading the figures

Every validation panel has one row per illuminant (D65, D50, HP1, E, A,
LED-B1).  In a swatch column, the **top half is the spectral reference** and
the **bottom half is the model**, one band per bounce from left to right.  The
rightmost plot is Delta-E 2000 against bounce index for every pipeline.  The
fluorescence panels also carry an "R-only" curve, the spectral reference with
fluorescence switched off, which shows how much of the signal comes from
fluorescence.

## Notes

* The two validation scripts use **different bases**:
  reflectance uses CIE 2006 XYZ on 380-780 nm; fluorescence uses the XYZU basis
  on 300-799 nm because the U channel needs UV support.  Numbers are not
  comparable across the two.
* They also **clip differently before Delta-E**: fluorescence clips linear
  sRGB at 0 only; reflectance clips to [0, 1].  In the reflectance panels the
  Delta-E curve therefore collapses once both reference and model saturate.
  See doc section 8.
* Jakob & Hanika upsampling is fitted on the visible range; the fluorescence
  script evaluates it down to 300 nm, which is a smooth extrapolation.
* `validate_reflectance.py` was converted from a notebook; one sRGB matrix
  constant was corrected to the IEC 61966-2-1 value.  The effect is below
  display precision.
* All scripts resolve `data/` and `results/` relative to their own location
  (`scripts/_paths.py`); they run from any working directory.

## Related repositories

* `pseudo-spectral-godot` -- the Godot 4.6 implementation this repository
  validates (`addons/fluorescent_materials/`).  The constants printed by
  `generate_godot_matrices.py` are pasted into
  `fluorescent_full.gdshader` and `fluorescent_controller.gd`.

## References

* L. Belcour, A. Fichet, P. Barla. *A Fluorescent Material Model for
  Non-Spectral Editing & Rendering.* SIGGRAPH 2025. hal-05267431.
* A. Fichet, L. Belcour, P. Barla. *Non-Orthogonal Reduction for Rendering
  Fluorescent Materials in Non-Spectral Engines.* CGF 2024.
* W. Jakob, J. Hanika. *A Low-Dimensional Function Space for Efficient
  Spectral Upsampling.* CGF (EG) 2019.
