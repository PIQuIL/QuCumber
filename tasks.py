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

import os
from datetime import datetime

from invoke import task

ALLOWED_BRANCHES = ["master", "develop"]


# Master should rebuild *everything*
# tags/other branches should only build themselves

@task
def build_docs(c):
    old_ref = c.run("git rev-parse HEAD").stdout.split('\n')[0].strip()

    head_name = c.run("git describe --all").stdout \
                 .split('\n')[0].strip() \
                 .split('/')[-1]

    all_refs = c.run("git tag --list", hide='out').stdout.split('\n')
    all_refs = [tag for tag in all_refs if tag]
    all_refs += ALLOWED_BRANCHES

    if head_name == "master":
        refs = [r for r in all_refs]  # copy all_refs
    else:
        refs = [head_name]

    with c.cd("./docs/"):
        c.run("mkdir -p _static _templates")
        c.run("make clean", hide='out')

        build_dirs = []
        for ref in refs:
            c.run("git checkout " + ref)

            b_dir = "_build/html/{}".format(ref)
            build_dirs += b_dir

            c.run("sphinx-build -b html ./ {} -aT".format(b_dir))

        if head_name == "master":
            c.run("touch _build/html/.nojekyll")
            c.run('echo \"<meta http-equiv=\\"refresh\\" content=\\"0; '
                  'url=./master/index.html\\" />\" > _build/html/index.html')

            for ref in refs:
                c.run('echo _build/html/{}/index.html '.format(ref)
                      + '>> _build/html/versions.html')

    c.run("git checkout gh-pages")

    dir_contents = os.listdir()
    dir_contents = ((set(dir_contents) - set(all_refs))
                    | (set(all_refs) - set(refs)))
    for item in dir_contents:
        if os.path.isfile(item):
            os.remove(item)
        elif os.path.isdir(item):
            os.rmdir(item)

    c.run("mv ./docs/_build/html/* ./")
    c.run("git add -A")
    c.run("git commit -m \"({}) - Deploy docs to GitHub Pages.\""
          .format(datetime.today().strftime("%Y-%m-%d"))

    # c.run("git push")

    # TODO: clear gh-pages file tree, and pop out build contents into root
    # then commit and push everything

    # c.run("git checkout {}".format(old_ref))  # revert to old_ref once done
