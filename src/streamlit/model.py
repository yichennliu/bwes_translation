import streamlit as st
import io
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

@st.cache(allow_output_mutation=True)
def load_vec(emb_path):
    vectors = []
    word2id = {}
    with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        next(f)
        for i, line in enumerate(f):
            word, vect = line.rstrip().split(' ', 1)
            vect = np.fromstring(vect, sep=' ')
            assert word not in word2id, 'word found twice'
            vectors.append(vect)
            word2id[word] = len(word2id)
    id2word = {v: k for k, v in word2id.items()}
    embeddings = np.vstack(vectors)
    return embeddings, id2word, word2id


# calculate nearest neighbour
def get_nn(word, src_emb, src_id2word, tgt_emb, tgt_id2word, K=5):
    results = []
    # print("Nearest neighbors of \"%s\":" % word)
    word2id = {v: k for k, v in src_id2word.items()}
    word_emb = src_emb[word2id[word]]
    scores = (tgt_emb / np.linalg.norm(tgt_emb, 2, 1)[:, None]).dot(word_emb / np.linalg.norm(word_emb))
    k_best = scores.argsort()[-K:][::-1]
    for i, idx in enumerate(k_best):
        # print('%.4f - %s' % (scores[idx], tgt_id2word[idx]))
        results.append((tgt_id2word[idx],scores[idx]))
    return results