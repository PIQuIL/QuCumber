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

from invoke import task, Collection
from invocations.pytest import test, coverage
from invocations.checks import blacken
from jinja2 import Environment, FileSystemLoader

##############################################################################
# --- Documentation Building -------------------------------------------------
##############################################################################

ALLOWED_BRANCHES = ["master", "develop"]
IGNORED_TAGS = ["v0.1.2", "v0.2.0"]

# Jinja stuff for making the versions.html page
versions_template = Environment(
    loader=FileSystemLoader("./docs/_templates")
).get_template("versions.html")


@task
def build_docs(c):
    old_ref = c.run("git rev-parse HEAD", hide="out").stdout.split("\n")[0].strip()

    head_name = (
        c.run("git describe --all", hide="out")
        .stdout.split("\n")[0]
        .strip()
        .split("/")[-1]
    )

    all_refs = c.run("git tag --list", hide="out").stdout.split("\n")
    all_refs = [tag for tag in all_refs if tag and tag not in IGNORED_TAGS]
    all_refs = ALLOWED_BRANCHES + sorted(all_refs)

    if head_name == "master":
        refs = [r for r in all_refs]  # copy all_refs
    else:
        refs = [head_name]

    with c.cd("./docs/"):
        c.run("mkdir -p _static _templates", hide="out")
        c.run("make clean", hide="out")

        for ref in refs:
            c.run("git checkout -q " + ref, echo=True)

            c.run(
                "sed -i 's/^__version__.*$/__version__ = \"{}\"/g' ".format(ref)
                + "../qucumber/__version__.py",
                echo=True,
            )

            b_dir = "_build/html/{}".format(ref)

            c.run("pip install -e ../", echo=True)
            c.run("python -m sphinx -b html ./ {} -aqT".format(b_dir), echo=True)
            c.run("pip uninstall -y qucumber", echo=True)
            c.run("git checkout -- ../qucumber/__init__.py", echo=True)

        c.run("touch _build/html/.nojekyll", echo=True)

        if head_name == "master":
            c.run("git checkout -q master", echo=True)
            c.run("cp _templates/index.html _build/html/index.html", echo=True)
            with open("./docs/_build/html/versions.html", "w") as f:
                f.write(versions_template.render(refs=[ref for ref in refs]))

    # revert back to old_ref once done
    c.run("git checkout -q " + old_ref, echo=True)


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
        raise RuntimeError(f"License Header missing in {num_fails} files.")
    else:
        print("License checking completed successfully.")


##############################################################################
# --- PyTest -----------------------------------------------------------------
##############################################################################


@task
def pytest(c, k=None, coverage=False, show_output=True):
    flags = []
    if show_output:
        flags.append("-s")
    if k is not None:
        flags.append("-k '{}'".format(k))
    if coverage:
        flags.append("--cov=qucumber")
        flags.append("--no-cov-on-fail")
        flags.append("--cov-report=term-missing")

    c.run("pytest " + " ".join(flags), pty=True)


##############################################################################
# --- Code Formatting --------------------------------------------------------
##############################################################################


@task
def flake8(c):
    c.run("flake8")


@task
def blacken_alt(c):
    c.run("black ./")


ns = Collection(build_docs, license_check, flake8, blacken, test, coverage)
