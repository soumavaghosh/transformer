import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from trans_encoder_unit import encoder_unit

class trans_encoder(nn.Module):

    """
        Creates a stack of encoder unit. Need to specify number of stacking units as arguements.
        Returns the encoder output for each unit of the sequence
    """

    def __init__(self, n=6):
        super(trans_encoder, self).__init__()

        self.unit = n
        self.stk = nn.ModuleList()

        for _ in range(self.unit):
            self.stk.append(encoder_unit())

    def forward(self, emb):

        for i in range(self.unit):
            emb = self.stk[i](emb)

        return(emb)