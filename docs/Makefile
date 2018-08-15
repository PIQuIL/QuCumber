# Minimal makefile for Sphinx documentation

# You can set these variables from the command line.
SPHINXOPTS    = -j auto -aT
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = QuCumber
SOURCEDIR     = .
BUILDDIR      = _build
PYTHONPATH    = ../


# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

test:
	@$(SPHINXBUILD) -nW -b dummy "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

livehtml:
	sphinx-autobuild \
		-z ../qucumber \
		-z ../examples \
		-i *.swp \
		-r '_examples/*' \
		-i Makefile \
		-b html \
		$(SPHINXOPTS) $(SOURCEDIR) $(BUILDDIR)/html

livetest:
	sphinx-autobuild -nW \
		-z ../qucumber \
		-z ../examples \
		-i *.swp \
		-r '_examples/*' \
		-i Makefile \
		-b dummy \
		$(SPHINXOPTS) $(SOURCEDIR) $(BUILDDIR)
