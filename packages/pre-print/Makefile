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

sync-figures:
	@mkdir -p $(IMGDIR)
	@updated=0; \
	for f in $(SRCIMGS)/*.png; do \
		dest=$(IMGDIR)/$$(basename "$$f"); \
		if [ ! -f "$$dest" ] || ! cmp -s "$$f" "$$dest"; then \
			cp "$$f" "$$dest"; \
			updated=$$((updated + 1)); \
			echo "  updated: $$(basename $$f)"; \
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
