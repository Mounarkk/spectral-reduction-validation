# Spectral-reduction validation -- top-level targets.
#   make venv          create .venv and install pinned dependencies
#   make check         machine-precision checks of the Godot port (exit 1 on failure)
#   make all           check + every validation figure
#   make doc           build doc/main.pdf (needs latexmk; or upload doc/ to Overleaf)

PY ?= .venv/bin/python

.PHONY: all venv check matrices reflectance fluorescence upscale doc figures clean clean-results

all: check reflectance fluorescence upscale

venv:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

check:
	$(PY) scripts/verify_fbar_convention.py
	$(PY) scripts/verify_godot_shader.py

matrices:
	$(PY) scripts/generate_godot_matrices.py

reflectance:
	$(PY) scripts/validate_reflectance.py

fluorescence:
	$(PY) scripts/validate_fluorescence.py

upscale:
	$(PY) scripts/fig_upscale_diag.py

figures:
	$(MAKE) -C doc figures

doc: figures
	$(MAKE) -C doc

clean:
	rm -rf scripts/__pycache__
	$(MAKE) -C doc clean

clean-results:
	rm -rf results/fluorescence results/reflectance results/upscale_diag
