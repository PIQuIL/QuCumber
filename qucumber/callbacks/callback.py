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


class Callback:
    """Base class for callbacks."""

    def on_train_start(self, rbm):
        """Called at the start of the training cycle.

        :param rbm: The RBM being trained.
        :type rbm: BinomialRBM
        """
        pass

    def on_train_end(self, rbm):
        """Called at the end of the training cycle.

        :param rbm: The RBM being trained.
        :type rbm: BinomialRBM
        """
        pass

    def on_epoch_start(self, rbm, epoch):
        """Called at the start of each epoch.

        :param rbm: The RBM being trained.
        :type rbm: BinomialRBM
        :param epoch: The current epoch.
        :type epoch: int
        """
        pass

    def on_epoch_end(self, rbm, epoch):
        """Called at the end of each epoch.

        :param rbm: The RBM being trained.
        :type rbm: BinomialRBM
        :param epoch: The current epoch.
        :type epoch: int
        """
        pass

    def on_batch_start(self, rbm, epoch, batch):
        """Called at the start of each batch.

        :param rbm: The RBM being trained.
        :type rbm: BinomialRBM
        :param epoch: The current epoch.
        :type epoch: int
        :param batch: The current batch index.
        :type batch: int
        """
        pass

    def on_batch_end(self, rbm, epoch, batch):
        """Called at the end of each batch.

        :param rbm: The RBM being trained.
        :type rbm: BinomialRBM
        :param epoch: The current epoch.
        :type epoch: int
        :param batch: The current batch index.
        :type batch: int
        """
        pass
