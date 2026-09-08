# Technical companion document.
#   make            build main.pdf with latexmk
#   make figures    copy the figures the document embeds from ../results
#   make clean      remove build artefacts (keeps main.pdf)
#
# No local LaTeX?  Upload this folder (main.tex + figures/) to Overleaf as-is.

RES = ../results
FIG = figures

FIGS = \
	$(FIG)/reflectance_warm.png \
	$(FIG)/reflectance_red.png \
	$(FIG)/fluorescence_m2_blue.png \
	$(FIG)/fluorescence_m2_blue_diag.png \
	$(FIG)/fluorescence_m2_blue_param_sweep.png \
	$(FIG)/fluorescence_u_channel.png \
	$(FIG)/upscale_de_chroma_spectra.png

.PHONY: all figures clean

all: main.pdf

main.pdf: main.tex
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

figures: $(FIGS)

$(FIG)/reflectance_warm.png:                 $(RES)/reflectance/warm.png
	cp $< $@
$(FIG)/reflectance_red.png:                  $(RES)/reflectance/red.png
	cp $< $@
$(FIG)/fluorescence_m2_blue.png:             $(RES)/fluorescence/method2/blue.png
	cp $< $@
$(FIG)/fluorescence_m2_blue_diag.png:        $(RES)/fluorescence/method2/blue_diag_D65.png
	cp $< $@
$(FIG)/fluorescence_m2_blue_param_sweep.png: $(RES)/fluorescence/method2/blue_param_sweep_D65_M2.png
	cp $< $@
$(FIG)/fluorescence_u_channel.png:           $(RES)/fluorescence/u_channel_comparison.png
	cp $< $@
$(FIG)/upscale_de_chroma_spectra.png:        $(RES)/upscale_diag/de_chroma_spectra.png
	cp $< $@

clean:
	latexmk -c
