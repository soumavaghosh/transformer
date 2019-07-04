from nltk import word_tokenize
from trans_wrd_embed import trans_word_emb
from enc_output import encoder_output
from torch import nn
import torch.optim as optim
from random import randint, shuffle
import torch
from enc_stack import trans_encoder

f = open('imdb_labelled.txt', 'rb')
data = f.readlines()

data = [x.decode("utf-8") for x in data]
txt = [x.split('\t')[0].strip().lower() for x in data]
txt = [word_tokenize(x) for x in txt]
label = [int(x.split('\t')[1].strip().replace('\n', '')) for x in data]

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

final_data = [(txt_label[i], label[i]) for i in range(len(label))]

print('done')

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
losses = []

epoch = 20
for _ in range(epoch):
    opt.zero_grad()

    shuffle(final_data)
    train_data = final_data[:batch]

    txt_data = [x[0] for x in train_data]
    lab_data = [x[1] for x in train_data]
    lab = torch.tensor(lab_data, dtype=torch.long, requires_grad=False)

    emb = embed(txt_data, pos_data)
    out = encoder(emb)
    prob = enc_conv(out)

    loss = loss_f(prob, lab)
    loss.backward()
    losses.append(loss)

    embed.word_emb.weight.grad[0].zero_()
    opt.step()
    print(loss)