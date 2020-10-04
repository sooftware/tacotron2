import torch
import torch.nn as nn


class TextToSpeechCriterion(nn.Module):
    def __init__(self):
        super(TextToSpeechCriterion, self).__init__()

    def forward(self, output, targets):
        with torch.no_grad():
            feat_outputs = output["feat_outputs"]
            # TODO : Criterion Implementation
