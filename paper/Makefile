# Build the paper: regenerate the harness-derived tables and prompts, then compile with tectonic.
PY ?= ../.venv/bin/python

all: main.pdf

generated: gen_paper.py
	cd .. && $(PY:../%=%) paper/gen_paper.py

main.pdf: main.tex references.bib generated
	tectonic --keep-logs main.tex

clean:
	rm -f main.pdf main.log main.aux main.bbl main.blg main.out

.PHONY: all generated clean
