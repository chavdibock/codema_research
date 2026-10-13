# Analysis pipeline (spec §8.5). Run from the Experiments/ directory:
#   make -f experiments/analysis/Makefile all RESULTS=$$CDX_RESULTS
# Runs on the login node or locally after rsync; needs no GPU.
PYTHON  ?= python
RESULTS ?= $(or $(CDX_RESULTS),results)
CACHE   ?= $(or $(CDX_CACHE),cache)
SUMMARY ?= summary

.PHONY: all aggregate tables figures clean

all: aggregate tables figures

aggregate:
	$(PYTHON) -m experiments.analysis.aggregate --results $(RESULTS) --summary $(SUMMARY)

tables: aggregate
	$(PYTHON) -m experiments.analysis.tables --summary $(SUMMARY) --out $(SUMMARY)/tables

figures: aggregate
	$(PYTHON) -m experiments.analysis.figures --summary $(SUMMARY) --out $(SUMMARY)/figures --cache $(CACHE)

clean:
	rm -rf $(SUMMARY)
