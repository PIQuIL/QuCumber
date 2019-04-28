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

# ----------------------------------------------------------------------------
# Originally based on code from the `bokeh` github repo:
# https://github.com/bokeh/bokeh/blob/e693d61/bokeh/application/handlers/notebook.py
#
# Copyright (c) 2012 - 2018, Anaconda, Inc. All rights reserved.
#
# Powered by the Bokeh Development Team.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# ----------------------------------------------------------------------------


import re
import nbconvert
import logging

log = logging.getLogger(__name__)


class StripMagicsProcessor(nbconvert.preprocessors.Preprocessor):
    """
    Preprocessor to convert notebooks to Python source while stripping
    out all magics (i.e. IPython specific syntax).
    """

    _magic_pattern = re.compile(r"^\s*(?P<magic>%+\w\w+)($|(\s+))")

    def strip_magics(self, source):
        """
        Given the source of a cell, filter out all cell and line magics.
        """
        filtered = []
        for line in source.splitlines():
            match = self._magic_pattern.match(line)
            if match is None:
                filtered.append(line)
            else:
                msg = "[NbConvertApp.Preprocessor] Stripping out IPython magic {magic} in code cell {cell}"
                message = msg.format(
                    cell=self._cell_counter, magic=match.group("magic")
                )
                log.warning(message)
                s = line.replace(match[0], "").strip()
                if s:
                    filtered.append(s)
        return "\n".join(filtered)

    def preprocess_cell(self, cell, resources, index):
        if cell["cell_type"] == "code":
            self._cell_counter += 1
            cell["source"] = self.strip_magics(cell["source"])
        return cell, resources

    def __call__(self, nb, resources):
        self._cell_counter = 0
        return self.preprocess(nb, resources)
