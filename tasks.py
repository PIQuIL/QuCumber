
from invoke import task

import subprocess

ALLOWED_BRANCHES = ["master", "develop"]
IGNORED_TAGS = ["v0.1.2", "v0.2.0"]


def get_tags():
    s = subprocess.check_output(["git", "tag", "--list"],
                                universal_newlines=True)
    tags = [tag for tag in s.split('\n') if tag and tag not in IGNORED_TAGS]
    return tags


def list_important_refs():
    return get_tags() + ALLOWED_BRANCHES


@task
def build(c, ref_name):
    with c.cd("./docs/"):
        with c.prefix("VERSION={} && SPHINXOPTS=-aT".format(ref_name)):
            c.run("echo $VERSION")
            c.run("make gh-pages")


@task
def build_all(c):
    refs = c.run("git tag --list", hide='out').stdout.split('\n')
    refs = [tag for tag in refs if tag and tag not in IGNORED_TAGS]
    refs += ALLOWED_BRANCHES

    with c.cd("./docs/"):
        c.run("mkdir -p _static _templates")
        c.run("make clean", hide='out')

    for ref in refs:
        print()
        c.run("git checkout " + ref)
        build(c, ref)

    c.run("git checkout master")
