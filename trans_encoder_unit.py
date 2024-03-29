import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class encoder_unit(nn.Module):
    """
        Creates a each unit of the encoder. Consists of the self attention block and the fully connected NN.
        Takes in the embedded input and outputs the encoded output
    """

    def __init__(self, embedding_size=512, attn_head=8):
        super(encoder_unit, self).__init__()

        self.embedding_size = embedding_size
        self.attn_head = attn_head

        self.q_list = nn.ModuleList()
        self.k_list = nn.ModuleList()
        self.v_list = nn.ModuleList()

        for _ in range(attn_head):
            self.q_list.append(nn.Linear(embedding_size, int(embedding_size / attn_head)))
            self.k_list.append(nn.Linear(embedding_size, int(embedding_size / attn_head)))
            self.v_list.append(nn.Linear(embedding_size, int(embedding_size / attn_head)))

        self.w0 = nn.Linear(embedding_size, embedding_size)

        self.l_norm1 = nn.LayerNorm(embedding_size)

        self.w1 = nn.Linear(embedding_size, embedding_size*4, bias=True)
        self.w2 = nn.Linear(embedding_size*4, embedding_size)

        self.l_norm2 = nn.LayerNorm(embedding_size)

    def forward(self, emb):

        q_tmp = []
        k_tmp = []
        v_tmp = []
        z_tmp = []

        for i in range(len(self.q_list)):
            q_tmp.append(self.q_list[i](emb))
            k_tmp.append(self.k_list[i](emb))
            v_tmp.append(self.v_list[i](emb))
            z = torch.bmm(q_tmp[-1], torch.transpose(k_tmp[-1], 1, 2))
            weight = F.softmax(z, dim=2)
            z = torch.bmm(weight, v_tmp[-1])/np.sqrt(self.embedding_size / self.attn_head)
            z_tmp.append(z)

        z = z_tmp[0]
        for i in z_tmp[1:]:
            z = torch.cat((z, i), 2)

        z = self.w0(z)

        out1 = z + emb
        out1 = self.l_norm1(out1)

        out2 = F.relu(self.w1(out1))
        out2 = self.w2(out2)
        out2 = out1 + out2
        out2 = self.l_norm2(out2)

        return out2
