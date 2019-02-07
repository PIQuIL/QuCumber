# Copyright 2018 PIQuIL - All Rights Reserved

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import os.path
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

version_file = {}
with open(os.path.join("qucumber", "__version__.py"), "r") as f:
    exec(f.read(), version_file)

install_requires = [
    "torch==1.0; sys_platform != 'win32'",
    "tqdm>=4.23",
    "numpy>=1.13",
    "matplotlib>=2.2",
]

# because RTD runs out of memory when using `pip install -e .[rtd]` to install
# docs dependencies for some reason
with open(".build_tools/readthedocs/requirements.txt", "r") as reqs:
    rtd_requires = [line.strip() for line in reqs.readlines()]

doc_requires = rtd_requires + ["sphinx_rtd_theme>=0.4.1", "sphinx-autobuild>=0.7.1"]

build_requires = ["setuptools>=40.0.0", "wheel>=0.31.1"]

test_requires = ["pytest>=3.7.1", "tox>=3.2.1"]

coverage_requires = test_requires + ["pytest-cov>=2.5.1"]

style_requires = [
    "radon>=2.2.0",
    "black==18.6b4; python_version>='3.6'",
    "flake8>=3.7.5",
    "flake8-per-file-ignores>=0.6",
    "flake8-bugbear>=18.2.0",
]

travis_requires = (
    build_requires
    + coverage_requires
    + style_requires
    + ["invoke>=1.1.1", "nbconvert>=5.3.1"]
)

appveyor_requires = build_requires + test_requires

dev_requires = travis_requires + doc_requires + ["pre_commit>=1.10.5", "nbval>=0.9.1"]

extras_require = {
    "dev": dev_requires,
    "test": test_requires,
    "coverage": coverage_requires,
    "style": style_requires,
    "travis": travis_requires,
    "appveyor": appveyor_requires,
    "rtd": rtd_requires,
    "doc": doc_requires,
}

setuptools.setup(
    name="qucumber",
    version=version_file["__version__"],
    description="Neural Network Quantum State Tomography.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: Implementation :: CPython",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.5,<4",
    install_requires=install_requires,
    extras_require=extras_require,
    include_package_data=True,
    url="http://github.com/PIQuIL/QuCumber",
    author="PIQuIL",
    author_email="piquildbeets@gmail.com",
    license="Apache License 2.0",
    packages=setuptools.find_packages(exclude=("examples", "docs", "tests")),
    zip_safe=False,
)
