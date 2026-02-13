TEX      = pdflatex
BIB      = bibtex
MAIN     = ms
IMGDIR   = figures/img
SRCIMGS  = ../docs/static/img/results

.PHONY: all clean distclean link-figures

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex $(MAIN).bib $(wildcard figures/tikz/*.tex)
	$(TEX) $(MAIN)
	$(BIB) $(MAIN)
	$(TEX) $(MAIN)
	$(TEX) $(MAIN)

link-figures:
	@mkdir -p $(IMGDIR)
	@for f in $(SRCIMGS)/*.png; do \
		ln -sf ../../$$f $(IMGDIR)/$$(basename "$$f"); \
	done
	@echo "Linked $$(ls $(IMGDIR)/*.png 2>/dev/null | wc -l | tr -d ' ') figures"

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out
	rm -f $(MAIN).fdb_latexmk $(MAIN).fls $(MAIN).synctex.gz
	rm -f $(MAIN).toc $(MAIN).lof $(MAIN).lot $(MAIN).run.xml $(MAIN).bcf

distclean: clean
	rm -f $(MAIN).pdf
	rm -rf $(IMGDIR)
