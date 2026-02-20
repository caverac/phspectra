TEX      = pdflatex
BIB      = bibtex
MAIN     = ms
SRCIMGS  = ../docs/static/img/results

.PHONY: all clean distclean sync-figures submission

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex $(MAIN).bib $(wildcard *.tex) $(wildcard *.png)
	$(TEX) $(MAIN)
	$(BIB) $(MAIN)
	$(TEX) $(MAIN)
	$(TEX) $(MAIN)

FIGURES = water-level-stages.png synthetic-f1.png rms-scatter.png \
          grs-map-plot.png velocity-spacing-plot.png

sync-figures:
	@updated=0; \
	for f in $(FIGURES); do \
		src=$(SRCIMGS)/$$f; \
		if [ ! -f "$$src" ]; then \
			echo "  missing: $$f"; \
			continue; \
		fi; \
		if [ ! -f "$$f" ] || ! cmp -s "$$src" "$$f"; then \
			cp "$$src" "$$f"; \
			updated=$$((updated + 1)); \
			echo "  updated: $$f"; \
		fi; \
	done; \
	echo "Synced $$updated figure(s)"

ZIPDIR   = /tmp/phspectra
ZIPFILE  = $(ZIPDIR)/phspectra-preprint.zip

submission:
	@mkdir -p $(ZIPDIR)
	@zip -q $(ZIPFILE) $(MAIN).tex $(MAIN).bib $(MAIN).pdf *.png persistence-diagram.tex pipeline-overview.tex water-level.tex
	@echo "Created $(ZIPFILE)"

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out
	rm -f $(MAIN).fdb_latexmk $(MAIN).fls $(MAIN).synctex.gz
	rm -f $(MAIN).toc $(MAIN).lof $(MAIN).lot $(MAIN).run.xml $(MAIN).bcf

distclean: clean
	rm -f $(MAIN).pdf
	rm -f $(FIGURES)
