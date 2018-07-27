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

def load_target_psi(N,path_to_target_psi):
    psi_data = np.loadtxt(path_to_target_psi)
    D = 2**N#int(len(psi_data)/float(len(bases)))
    psi=torch.zeros(2,D, dtype=torch.double)
    if (len(psi_data.shape)<2):
        psi[0] = torch.tensor(psi_data,dtype=torch.double)
        psi[1] = torch.zeros(D,dtype=torch.double)
    else:
        psi_real = torch.tensor(psi_data[0:D,0],dtype=torch.double)
        psi_imag = torch.tensor(psi_data[0:D,1],dtype=torch.double)
        psi[0]   = psi_real
        psi[1]   = psi_imag
        
    return psi 
















