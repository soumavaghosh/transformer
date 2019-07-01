from nltk import word_tokenize
from trans_encoder_unit import encoder_unit
from trans_wrd_embed import trans_word_emb
from enc_output import encoder_output
from torch import nn
import torch.optim as optim
from random import randint
import torch

f = open('imdb_labelled.txt', 'rb')
data = f.readlines()

data = [x.decode("utf-8") for x in data]
txt = [x.split('\t')[0].strip().lower() for x in data]
txt = [word_tokenize(x) for x in txt]
label = [int(x.split('\t')[1].strip().replace('\n', '')) for x in data]
lab = torch.tensor(label, dtype=torch.long, requires_grad=False).unsqueeze(1)

words = []
max_len = 0
for x in txt:
    if len(x) > max_len:
        max_len = len(x)
    words.extend(x)
    words = list(set(words))

words = sorted(words)

word_to_id = {w:i+1 for i, w in enumerate(words)}

def padding(x):
    global max_len
    l=0
    if len(x)<max_len:
        l = max_len-len(x)
    x.extend([0]*l)

    return(x)

txt_label = [[word_to_id[i]for i in x] for x in txt]

txt_label = [padding(x) for x in txt_label]

# ----------------------------------------------------------

embed = trans_word_emb(len(word_to_id)+1, max_len)
encoder = encoder_unit(len(word_to_id)+1, max_len)
enc_conv = encoder_output(max_len)

loss_f = nn.CrossEntropyLoss()
params = list(embed.parameters()) + list(encoder.parameters()) + list(enc_conv.parameters())

embed.train()
encoder.train()
enc_conv.train()

opt = optim.SGD(params, lr = 0.001)

for i in range(10):

    opt.zero_grad()
    pos = list(range(max_len))
    ind = randint(0, len(label))

    x_word_emb, x_pos_emb = embed(txt_label[ind], pos)
    out = encoder(x_word_emb, x_pos_emb)
    prob = enc_conv(out).unsqueeze(0)

    loss = loss_f(prob, lab[ind])
    loss.backward()
    opt.step()
    print(loss)