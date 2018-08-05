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

LENGTH_CUTOFF = 15
EXCLUDED_FILES = ["setup.py"]
EXTENSIONS = [".py"]

with open('LICENSE_HEADER', 'r') as lh:
    header = [line.strip() for line in lh.readlines() if line.strip()]


def check_license(file_path):
    with open(file_path, 'r') as f:
        file_contents = f.read().replace('\n', '')
        f.seek(0)  # go back to start of file
        file_len = len(f.readlines())

    if file_len < LENGTH_CUTOFF:
        return 0

    if any(excl in file_path for excl in EXCLUDED_FILES):
        return 0

    if any(line not in file_contents for line in header):
        print("License Header missing in file: " + file_path)
        return 1

    return 0


if __name__ == '__main__':
    num_fails = 0

    paths = chain(*[pathlib.Path('.').glob("**/*" + extension)
                    for extension in EXTENSIONS])

    for path in paths:
        num_fails += check_license(str(path))

    if num_fails > 0:
        raise RuntimeError(f"License Header missing in {num_fails} files.")
    else:
        print("License checking completed successfully.")
