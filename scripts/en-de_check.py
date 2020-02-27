#!/usr/bin/python3
import io
import itertools
import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import sys
import re
import string
import subprocess

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


src_path = '../../../vecmap/SRC_MAPPED-en-de-identical.EMB'
tgt_path = '../../../vecmap/TRG_MAPPED-en-de-identical.EMB'

src_embeddings, src_id2word, src_word2id = load_vec(src_path)
tgt_embeddings, tgt_id2word, tgt_word2id = load_vec(tgt_path)


# calculate nearest neighbour
def get_nn(word, src_emb, src_id2word, tgt_emb, tgt_id2word, K=5):
    results = []
    print("Nearest neighbors of \"%s\":" % word)
    word2id = {v: k for k, v in src_id2word.items()}
    word_emb = src_emb[word2id[word]]
    scores = (tgt_emb / np.linalg.norm(tgt_emb, 2, 1)[:, None]).dot(word_emb / np.linalg.norm(word_emb))
    k_best = scores.argsort()[-K:][::-1]
    for i, idx in enumerate(k_best):
        print('%.4f - %s' % (scores[idx], tgt_id2word[idx]))
        results.append(tgt_id2word[idx])
    return results


# create PCA
def plot_similar_word(src_words, src_word2id, src_emb, tgt_words, tgt_word2id, tgt_emb, pca):
    Y = []
    word_labels = []
    for sw in src_words:
        try:
            Y.append(src_emb[src_word2id[sw]])
            word_labels.append(sw)
        except KeyError:
            print(sw + " not in source dictionary")
            continue

    for tw in tgt_words:
        try:
            Y.append(tgt_emb[tgt_word2id[tw]])
            word_labels.append(tw)

        except KeyError:
            print(tw + " not in target dictionary")

    # find tsne coords for 2 dimensions
    Y = pca.transform(Y)
    x_coords = Y[:, 0]
    y_coords = Y[:, 1]

    # display scatter plot
    plt.figure(figsize=(10, 8), dpi=80)
    plt.scatter(x_coords, y_coords, marker='x')

    for k, (label, x, y) in enumerate(zip(word_labels, x_coords, y_coords)):
        color = 'blue' if k < len(src_words) else 'red'  # src words in blue / tgt words in red
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points', fontsize=19,
                     color=color, weight='bold')

    plt.xlim(x_coords.min() - 0.2, x_coords.max() + 0.2)
    plt.ylim(y_coords.min() - 0.2, y_coords.max() + 0.2)
    plt.title('Visualization of the multilingual word embedding space')

    plt.show()


# extract all words in the file
def unique_words_from_file(fpath):
    with open(fpath, "r") as f:
        return set(itertools.chain.from_iterable(map(str.split, f)))


# save the results into csv file
def csv_output(rows):
    for count in range(1,3):
        csv_file_name = "output_" + str(count) + ".csv"
        with open(csv_file_name, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        f.close()


def remove_digit_punct_stopword(infile, outfile, lang):
    with open(outfile, 'w') as f:
        for line in open(infile):
            # split into words
            extract_words = line.split()
            for word in extract_words:
                filtered_word = re.sub('[^a-zA-Z \s\d+]', '', word)
                f.write(filtered_word + "\n")



if __name__ == '__main__':

    f = open('../../../output2.txt', 'w')
    # pca = PCA(n_components=2, whiten=True)
    # pca.fit(np.vstack([src_embeddings, tgt_embeddings]))
    # print('Variance explained: %.2f' % pca.explained_variance_ratio_.sum())

    src_vocab = unique_words_from_file("../../../de_freq")
    list_src_vocab = list(src_vocab)

    trg_vocab = unique_words_from_file("../../../en_freq")
    list_trg_vocab = list(trg_vocab)



    # split sentence into single words
    src_words = ['approximately','how', 'much', 'you', 'weigh']
    tgt_words = ['ungefähr','wie', 'viel', 'wiegen', 'sie']
    index = 0
    # spell = SpellChecker()
    matched = 0
    unmatched = 0
    # check mistranslations
    while (index < len(src_words)):
        for sw in src_words:
            try:
                # search top 10 translations
                words = get_nn(sw, src_embeddings, src_id2word, tgt_embeddings, tgt_id2word, K=10)
            except KeyError:
                print("\""+sw+"\"" + " not found in dictionary.", file=f)

            try:
                # search if the given translation is in generated results
                assert tgt_words[index] in words
                print("\""+tgt_words[index]+"\""+ " is a matched word.", file=f)
                matched += 1
            except AssertionError:
                print("\""+tgt_words[index]+"\""+ " not in search results.", file=f)
                unmatched += 1
                # print("Please check if the word is misspelled. List of Suggestion:", file=f)
                # print(list(spell.candidates(tgt_words[index])), file =f)

            index += 1

    print("Matched words" + " "+str(matched), file=f)
    print("Unmatched words" + " "+str(unmatched), file=f)

    f.close()

    #plot_similar_word(list_src_vocab, src_word2id, src_embeddings, list_trg_vocab, tgt_word2id, tgt_embeddings, pca)

