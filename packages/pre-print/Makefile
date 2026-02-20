TEX      = pdflatex
BIB      = bibtex
MAIN     = ms
IMGDIR   = figures/img
SRCIMGS  = ../docs/static/img/results

.PHONY: all clean distclean sync-figures

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex $(MAIN).bib $(wildcard figures/tikz/*.tex) $(wildcard figures/img/*.png)
	$(TEX) $(MAIN)
	$(BIB) $(MAIN)
	$(TEX) $(MAIN)
	$(TEX) $(MAIN)

FIGURES = water-level-stages.png synthetic-f1.png rms-scatter.png \
          grs-map-plot.png velocity-spacing-plot.png

sync-figures:
	@mkdir -p $(IMGDIR)
	@updated=0; \
	for f in $(FIGURES); do \
		src=$(SRCIMGS)/$$f; \
		dest=$(IMGDIR)/$$f; \
		if [ ! -f "$$src" ]; then \
			echo "  missing: $$f"; \
			continue; \
		fi; \
		if [ ! -f "$$dest" ] || ! cmp -s "$$src" "$$dest"; then \
			cp "$$src" "$$dest"; \
			updated=$$((updated + 1)); \
			echo "  updated: $$f"; \
		fi; \
	done; \
	echo "Synced $$updated figure(s)"

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out
	rm -f $(MAIN).fdb_latexmk $(MAIN).fls $(MAIN).synctex.gz
	rm -f $(MAIN).toc $(MAIN).lof $(MAIN).lot $(MAIN).run.xml $(MAIN).bcf

distclean: clean
	rm -f $(MAIN).pdf
	rm -rf $(IMGDIR)
