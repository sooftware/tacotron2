# -*- coding: utf-8 -*-
# Soohwan Kim @sooftware
# This source code is licensed under the Apache 2.0 License license found in the
# LICENSE file in the root directory of this source tree

import torch
import torch.nn as nn


class MelPredictNetworkCriterion(nn.Module):
    def __init__(self):
        super(MelPredictNetworkCriterion, self).__init__()

    def forward(self, output, targets):
        with torch.no_grad():
            feat_outputs = output["feat_outputs"]
            # TODO : Criterion Implementation
