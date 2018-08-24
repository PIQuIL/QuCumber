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

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

init_file = {}
with open("./qucumber/__version__.py", "r") as f:
    exec(f.read(), init_file)

with open("./requirements.txt", "r") as f:
    install_requires = [req.strip() for req in f.readlines()]

setuptools.setup(
    name="qucumber",
    version=init_file["__version__"],
    description="Neural Network Quantum State Tomography.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.5",
    install_requires=install_requires,
    include_package_data=True,
    url="http://github.com/PIQuIL/QuCumber",
    author="PIQuIL",
    author_email="piquildbeets@gmail.com",
    license="Apache License 2.0",
    packages=setuptools.find_packages(),
    zip_safe=False,
)
