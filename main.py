from nltk import word_tokenize
from trans_encoder_unit import encoder_unit
from trans_wrd_embed import trans_word_emb
from enc_output import encoder_output

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

# ----------------------------------------------------------

embed = trans_word_emb(len(word_to_id)+1, max_len)
encoder = encoder_unit(len(word_to_id)+1, max_len)
enc_conv = encoder_output(max_len)

for i in range(1):

    pos = list(range(max_len))

    x_word_emb, x_pos_emb = embed(txt_label[0], pos)
    out = encoder(x_word_emb, x_pos_emb)
    prob = enc_conv(out)

print(encoder)
