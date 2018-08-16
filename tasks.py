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
from pprint import pformat
from itertools import chain

from invocations.pytest import coverage, test
from invoke import Collection, task


##############################################################################
# --- License Checking -------------------------------------------------------
##############################################################################


def is_license_missing(file_path, length_cutoff, exclude):
    with open(file_path, "r") as f:
        file_contents = f.read().replace("\n", "")
        f.seek(0)  # go back to start of file
        file_len = len(f.readlines())

    if file_len < length_cutoff:
        return False

    if any(excl in file_path for excl in exclude):
        return False

    with open("LICENSE_HEADER", "r") as lh:
        header = [line.strip() for line in lh.readlines() if line.strip()]

    if any(line not in file_contents for line in header):
        print("License Header missing in file: " + file_path)
        return True

    return False


@task(iterable=["extensions", "exclude"])
def license_check(c, length_cutoff=15, extensions=None, exclude=None):
    """Make sure all python files with more than 15 lines of code contain the license header."""
    num_fails = 0

    extensions = extensions if extensions else [".py"]
    exclude = exclude if exclude else ["setup.py"]

    paths = chain(
        *[pathlib.Path(".").glob("**/*" + extension) for extension in extensions]
    )

    for path in paths:
        num_fails += int(is_license_missing(str(path), length_cutoff, exclude))

    if num_fails > 0:
        raise RuntimeError("License Header missing in {} files.".format(num_fails))
    else:
        print("License checking completed successfully.")


##############################################################################
# --- Notebook Linting -------------------------------------------------------
##############################################################################


@task(aliases=["lint_examples", "lint_notebooks"])
def lint_example_notebooks(c, linter="flake8"):
    """Lint notebooks in the `./examples` directory.

    Supports flake8 and black linters.

    :param linter: The linter to validate the notebooks with.
                   Can be one of: ["flake8", "black"]
    :type linter: str
    """
    to_script_command = "jupyter nbconvert {} --stdout --to script"
    linter_commands = {
        "black": "black --check --diff -",
        "flake8": "flake8 - --ignore=W391",  # ignore trailing newline
    }

    try:
        linter_command = linter_commands[linter]
    except KeyError:
        raise ValueError("Linter, {}, not supported!".format(linter))

    nb_paths = pathlib.Path("./examples").glob("**/*.ipynb")

    num_fails = 0
    failed_files = []

    for path in nb_paths:
        failed = False

        run = c.run(
            to_script_command.format(str(path)) + " | " + linter_command,
            warn=True,  # don't exit task on first fail
            echo=True,  # print bash command to stdout
        )
        if run.failed:
            num_fails += 1
            failed = True
        if failed:
            failed_files.append(str(path))

    if num_fails > 0:
        raise RuntimeError(
            "Notebook code isn't formatted properly.\n"
            + "Number of unformatted files reported: {}\n".format(num_fails)
            + "Files with errors: {}".format(pformat(failed_files))
        )


ns = Collection(license_check, test, coverage, lint_example_notebooks)
