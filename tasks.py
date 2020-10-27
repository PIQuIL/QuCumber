# Copyright 2019 PIQuIL - All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import pathlib
from pprint import pformat
from itertools import chain

from invoke import task, call
from invoke.exceptions import Exit


##############################################################################
# --- License Checking -------------------------------------------------------
##############################################################################


def is_license_missing(file_path, length_cutoff, exclude):
    if any(excl in file_path for excl in exclude):
        return False

    with open(file_path, "r") as f:
        file_contents = f.read().replace("\n", "")
        f.seek(0)  # go back to start of file
        file_len = len(f.readlines())

    if file_len < length_cutoff:
        return False

    with open("LICENSE_HEADER", "r") as lh:
        header = [line.strip() for line in lh.readlines() if line.strip()]

    if any(line not in file_contents for line in header):
        print("License Header missing in file: " + file_path)
        return True

    return False


@task(
    iterable=["extensions", "exclude"],
    help={
        "length-cutoff": (
            "The maximum length of a file can be without a license header. "
            "Default: 15"
        ),
        "extensions": (
            "File extensions to check for license headers. "
            "Can be provided multiple times. By default only checks "
            "files with a '.py' extension."
        ),
        "exclude": "Files to exclude. Can be provided multiple times.",
    },
)
def license_check(c, extensions, exclude, length_cutoff=15):
    """Make sure all python files with more than 15 lines of code contain the license header."""
    num_fails = 0

    extensions = set(extensions) | {".py"}
    exclude = set(exclude) | {".tox"}

    paths = chain(
        *[
            pathlib.Path(root).glob("**/*" + extension)
            for extension in extensions
            for root in [
                "./.build_tools",
                "./docs",
                "./examples",
                "./qucumber",
                "./tests",
            ]
        ]
    )

    for path in paths:
        num_fails += int(is_license_missing(str(path), length_cutoff, exclude))

    if num_fails > 0:
        raise Exit(
            message=f"License Header missing in {num_fails} files.", code=num_fails
        )
    else:
        print("License checking completed successfully.")


##############################################################################
# --- Notebook Linting -------------------------------------------------------
##############################################################################


@task(
    aliases=["lint_examples", "lint_notebooks"],
    help={
        "linter": "The linter to validate the notebooks with. Can be one of ['flake8', 'black']"
    },
)
def lint_example_notebooks(c, linter="flake8"):
    """Lint notebooks in the `./examples` directory."""
    to_script_command = (
        "jupyter nbconvert {} --stdout --to python "
        "--RegexRemovePreprocessor.patterns r'\\s*\\Z' "  # remove empty code cells
        "--Exporter.preprocessors strip_magics.StripMagicsProcessor "  # remove ipython magics
        "--template=code_cells_only.tpl "  # only lint code cells
        "| head -c -1"  # remove extra new-line at end
    )
    linter_commands = {
        "black": "black --check --diff -",
        "flake8": "flake8 - --show-source --extend-ignore=W391,W291,E402",
    }

    try:
        linter_command = linter_commands[linter]
    except KeyError:
        raise ValueError(f"Linter, {linter}, not supported!")

    nb_paths = pathlib.Path("./examples").glob("**/*[!checkpoint].ipynb")
    num_fails = 0
    failed_files = []

    for path in nb_paths:
        with c.cd("./.build_tools/invoke/"):
            run = c.run(
                "! "
                + to_script_command.format("../../" + str(path))
                + " | "
                + linter_command,
                warn=True,  # don't exit task on first fail
                echo=True,  # print generated bash command to stdout
            )
            if run.failed:
                num_fails += 1
                failed_files.append(str(path))

    if num_fails > 0:
        failed_files = sorted(failed_files)
        raise Exit(
            message=(
                "Notebook code isn't formatted properly "
                + f"(according to {linter}).\n"
                + f"Number of unformatted files reported: {num_fails}\n"
                + "Files with errors: {}".format(pformat(failed_files))
            ),
            code=num_fails,
        )


##############################################################################
# --- Full Style Check -------------------------------------------------------
##############################################################################


@task(
    pre=[call(license_check, (), ())],
    post=[
        call(lint_example_notebooks, linter="flake8"),
        call(lint_example_notebooks, linter="black"),
    ],
)
def style(c):
    """Runs all style/format checks on code."""
    num_fails = 0
    run = c.run("flake8 --extend-ignore=T", warn=True, echo=True)
    num_fails += int(run.failed)
    c.run("flake8 --select=T", warn=True, echo=True)
    run = c.run("black --diff --check .", warn=True, echo=True)
    num_fails += int(run.failed)

    if num_fails > 0:
        raise Exit(message="Code isn't formatted properly.", code=num_fails)
