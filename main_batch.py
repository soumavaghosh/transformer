from nltk import word_tokenize
from trans_wrd_embed import trans_word_emb
from enc_output import encoder_output
from torch import nn
import torch.optim as optim
from random import randint
import torch
from enc_stack import trans_encoder

f = open('imdb_labelled.txt', 'rb')
data = f.readlines()

data = [x.decode("utf-8") for x in data]
txt = [x.split('\t')[0].strip().lower() for x in data]
txt = [word_tokenize(x) for x in txt]
label = [int(x.split('\t')[1].strip().replace('\n', '')) for x in data]
lab = torch.tensor(label, dtype=torch.long, requires_grad=False)

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

embed = trans_word_emb(len(word_to_id)+1, max_len) # created embedding layer of words and position
encoder = trans_encoder(1) # contains a stack of encoders
enc_conv = encoder_output(max_len) # converts the final encoded outputs into final results

loss_f = nn.CrossEntropyLoss()
params = list(embed.parameters()) + list(encoder.parameters()) + list(enc_conv.parameters())

embed.train()
encoder.train()
enc_conv.train()

opt = optim.Adam(params, lr=0.005)

batch = 64
pos_data = []

pos = list(range(max_len))

for _ in range(batch):
    pos_data.append(pos)

# ----------------------------------------------------------
epoch = 20
for _ in range(epoch):
    opt.zero_grad()

    ind = randint(0, len(txt_label)-batch-1)
    emb = embed(txt_label[ind : ind+batch], pos_data)
    out = encoder(emb)
    prob = enc_conv(out)

    loss = loss_f(prob, lab[ind : ind+batch])
    loss.backward()
    embed.word_emb.weight.grad[0].zero_()
    opt.step()
    print(loss)