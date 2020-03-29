#!/usr/bin/python3
import csv
import io
import string
import sys
import numpy as np
from termcolor import colored

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


src_path = sys.argv[1]
tgt_path = sys.argv[2]

src_embeddings, src_id2word, src_word2id = load_vec(src_path)
tgt_embeddings, tgt_id2word, tgt_word2id = load_vec(tgt_path)


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
        results.append(tgt_id2word[idx])
    return results



if __name__ == '__main__':

    with open(sys.argv[3], 'r') as file:
        translation = []
        translated_words = []
        reference = []
        reference_words = []
        matching = {}
        csv_reader = csv.reader(file)
        next(csv_reader)
        for row in csv_reader:
            for sentence in row:
                sentence = sentence.split('\t')
                translation.append(sentence[1])
                reference.append(sentence[0])
                print("Generics: "+sentence[0])
                print("Translation: "+sentence[1])
                res = [s.translate(str.maketrans('', '', string.punctuation)) for s in sentence]
                result_list = [elem.lower() for elem in res]

                reference_words = result_list[0].split()
                matching = dict.fromkeys(reference_words, 0)

                translated_words = result_list[1].split()

                # check given translations
                matched = 0
                for sw in translated_words:
                    try:
                        # search top 10 translations
                        words = get_nn(sw, src_embeddings, src_id2word, tgt_embeddings, tgt_id2word, K=10)
                        for w in words:
                            if w in matching.keys():
                                matching[w] += 1

                    except KeyError:
                        print("\"" + sw + "\"" + " not found in dictionary.")

                for key, value in matching.items():
                    if value >= 1:
                        matched += 1

                precision = round(matched / len(translated_words), 2)
                recall = round(matched / len(reference_words),2)
                if precision < 0.5 or recall < 0.5:
                    print("Precision: "+ colored(str(precision), 'red'))
                    print("Recall: "+ str(recall))

                else:
                    print("Precision: " + str(precision))
                    print("Recall: " + str(recall))

