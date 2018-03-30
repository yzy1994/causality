#coding=utf-8
from gensim.models import Word2Vec
import numpy as np
import pickle

inputdir = './data/casual_input'

w2vmodel = Word2Vec.load('./data/Word60.model')

word2idx = {'<pad>':0}
vector_list = []
idx = 1

fopen = open(inputdir, 'r')
for line in fopen.readlines():
    contents = line.split('\t')
    source = contents[1].strip()
    target = contents[2].strip()
    words_list = []
    words_list.extend(source.split(' '))
    words_list.extend(target.split(' '))
    for word in words_list:
        if word not in word2idx:
            word2idx[word] = idx
            idx += 1

vocab = sorted(word2idx.items(),key = lambda x:x[1],reverse = False)
print len(vocab)
i = 0
for word in vocab:
    w_uni = word[0].decode('utf-8')
    if w_uni in w2vmodel.vocab:
        vector_list.append(w2vmodel[w_uni])
    else:
        i += 1
        vector_list.append(np.zeros(60 ,np.float32))

print i
print 'UNK IN VOCAB'


pickle.dump([word2idx, vector_list], open('./vocab.pkl', 'wb'))
