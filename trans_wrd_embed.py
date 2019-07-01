import torch
import torch.nn as nn

class trans_word_emb(nn.Module):

    def __init__(self, vocab_size, max_length, embedding_size=512):
        super(trans_word_emb, self).__init__()

        self.embedding_size = embedding_size
        self.vocab_size = vocab_size

        self.word_emb = nn.Embedding(vocab_size, embedding_size)
        self.pos_emb = nn.Embedding(max_length, embedding_size)

    def forward(self, input_data, pos_data):
        x = torch.tensor(input_data, dtype=torch.long, requires_grad=False)
        x_pos = torch.tensor(pos_data, dtype=torch.long, requires_grad=False)
        x_word_emb = self.word_emb(x)
        x_pos_emb = self.pos_emb(x_pos)

        return x_word_emb, x_pos_emb