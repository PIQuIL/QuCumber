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

import torch
import numpy as np


def load_data(tr_samples_path,tr_psi_path,tr_bases_path=None,bases_path=None):
    data = [] 
    data.append(torch.tensor(np.loadtxt(tr_samples_path, dtype= 'float32'),dtype=torch.double))
    target_psi_data = np.loadtxt(tr_psi_path, dtype= 'float32')
    target_psi = torch.zeros(2,len(target_psi_data), dtype=torch.double)
    target_psi[0] = torch.tensor(target_psi_data[:,0], dtype=torch.double)
    target_psi[1] = torch.tensor(target_psi_data[:,1], dtype=torch.double)
    data.append(target_psi)
    if tr_bases_path is not None:
        data.append(np.loadtxt(tr_bases_path,dtype=str))
    if bases_path is not None:
        bases_data = np.loadtxt(bases_path,dtype=str)
        bases = []
        for i in range(len(bases_data)):
            tmp = ""
            for j in range(len(bases_data[i])):
                if bases_data[i][j] is not " ":
                    tmp += bases_data[i][j]
            bases.append(tmp)
        data.append(bases)
    return data
    #return np.loadtxt(tr_samples_path, dtype= 'float32')
