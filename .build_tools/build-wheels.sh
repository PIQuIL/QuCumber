#!/bin/bash
set -e -x

# Install a system package required by our library
#yum install -y atlas-devel

PYBINS=(
  "/opt/python/cp35-cp35m/bin"
  "/opt/python/cp36-cp36m/bin"
  "/opt/python/cp37-cp37m/bin"
)

# Compile wheels
for PYBIN in ${PYBINS[@]}; do
    #"${PYBIN}/pip" install -r /io/requirements.txt
    "${PYBIN}/pip" wheel /io/ -w wheelhouse/
done

# Bundle external shared libraries into the wheels
#for whl in wheelhouse/*.whl; do
#    auditwheel repair "$whl" -w /io/wheelhouse/
#done

# Install packages and test
#for PYBIN in ${PYBINS[@]}; do
#    "${PYBIN}/pip" install python-manylinux-demo --no-index -f /io/wheelhouse
#    (cd "$HOME"; "${PYBIN}/nosetests" pymanylinuxdemo)
#done

