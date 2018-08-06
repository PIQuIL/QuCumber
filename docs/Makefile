# Minimal makefile for Sphinx documentation

# You can set these variables from the command line.
SPHINXOPTS    = -j auto -aT
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = QuCumber
SOURCEDIR     = .
BUILDDIR      = _build
PYTHONPATH    = ../

define move_examples
	rm -rf $(BUILDDIR)/../_examples
	mkdir -p $(BUILDDIR)/../_examples/
	find $(SOURCEDIR)/../examples -name '*.ipynb' \
		-not -path '$(SOURCEDIR)/../examples/.ipynb_checkpoints/*' \
		-exec cp -prv '{}' '$(BUILDDIR)/../_examples/' ';'
	rm -f $(BUILDDIR)/../_examples/*-checkpoint.ipynb
endef

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	$(call move_examples)
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

test:
	$(call move_examples)
	@$(SPHINXBUILD) -nW -b dummy "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

livehtml:
	$(call move_examples)
	sphinx-autobuild \
		-z ../qucumber \
		-i *.swp \
		-i Makefile \
		-b html \
		$(SPHINXOPTS) $(SOURCEDIR) $(BUILDDIR)/html

livetest:
	$(call move_examples)
	sphinx-autobuild -nW \
		-z ../qucumber \
		-i *.swp \
		-i Makefile \
		-b dummy \
		$(SPHINXOPTS) $(SOURCEDIR) $(BUILDDIR)

gh-pages:
	@$(call move_examples)
	if [ -z $(VERSION) ]; then\
		echo "Not a branch head/tagged release - skipping docs build.";\
	else\
		$(eval BUILDDIR=$(BUILDDIR)/html/$(VERSION))\
		$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)" -aT $(O);\
	fi

index:
	touch $(BUILDDIR)/html/.nojekyll
	echo "<meta http-equiv=\"refresh\" content=\"0; url=./master/index.html\" />" > $(BUILDDIR)/html/index.html
