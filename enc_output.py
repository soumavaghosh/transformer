import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class encoder_output(nn.Module):

    def __init__(self, max_len, embedding_size=512):
        super(encoder_output, self).__init__()
        self.embedding_size = embedding_size
        self.max_len = max_len
        self.w0 = nn.Linear(embedding_size, int(embedding_size/8))
        self.w1 = nn.Linear(int(embedding_size / 8), int(embedding_size / 64))
        self.w2 = nn.Linear(int(embedding_size / 64) * max_len, 256)
        self.w3 = nn.Linear(256, 8)
        self.w4 = nn.Linear(8, 2)

    def forward(self, enc_out):
        enc_out = self.w0(enc_out)
        enc_out = self.w1(enc_out)

        enc_flat = enc_out[0]
        for i in range(1,len(enc_out)):
            enc_flat = torch.cat((enc_flat, enc_out[i]), 0)

        enc_out = self.w2(enc_flat)
        enc_out = self.w3(enc_out)
        enc_out = F.softmax(self.w4(enc_out), dim=0)

        return enc_out