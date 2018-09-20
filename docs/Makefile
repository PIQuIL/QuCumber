# Minimal makefile for Sphinx documentation

# You can set these variables from the command line.
SPHINXOPTS    = -j 1 -aT
SPHINXAUTO    = \
	-z ../qucumber \
	-z ../examples \
	-i *.swp \
	-i **/.~*.ipynb \
	-i **/*-checkpoint.ipynb \
	-r '_examples/*' \
	-i Makefile
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = QuCumber
SOURCEDIR     = .
BUILDDIR      = _build
PYTHONPATH    = ../


# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile livehtml livetest

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

test:
	@$(SPHINXBUILD) -W -b dummy "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

livehtml:
	sphinx-autobuild -b html \
		$(SPHINXAUTO) \
		$(SPHINXOPTS) \
		"$(SOURCEDIR)" \
		"$(BUILDDIR)/html"

livetest:
	sphinx-autobuild -nW -b dummy \
		$(SPHINXAUTO) \
		$(SPHINXOPTS) \
		"$(SOURCEDIR)" \
		"$(BUILDDIR)"
