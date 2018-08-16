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


import pathlib
from itertools import chain

from invocations.pytest import coverage, test
from invoke import Collection, task


##############################################################################
# --- License Checking -------------------------------------------------------
##############################################################################


LENGTH_CUTOFF = 15
EXCLUDED_FILES = ["setup.py"]
EXTENSIONS = [".py"]


def is_license_missing(file_path):
    with open(file_path, "r") as f:
        file_contents = f.read().replace("\n", "")
        f.seek(0)  # go back to start of file
        file_len = len(f.readlines())

    if file_len < LENGTH_CUTOFF:
        return False

    if any(excl in file_path for excl in EXCLUDED_FILES):
        return False

    with open("LICENSE_HEADER", "r") as lh:
        header = [line.strip() for line in lh.readlines() if line.strip()]

    if any(line not in file_contents for line in header):
        print("License Header missing in file: " + file_path)
        return True

    return False


@task
def license_check(c):
    num_fails = 0

    paths = chain(
        *[pathlib.Path(".").glob("**/*" + extension) for extension in EXTENSIONS]
    )

    for path in paths:
        num_fails += int(is_license_missing(str(path)))

    if num_fails > 0:
        raise RuntimeError("License Header missing in {} files.".format(num_fails))
    else:
        print("License checking completed successfully.")


ns = Collection(license_check, test, coverage)
