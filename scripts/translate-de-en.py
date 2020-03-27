#!/usr/bin/python3
import csv
import io
import string
import sys
import numpy as np
import pandas as pd


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


src_path = '../../../vecmap/SRC_MAPPED-de-en-identical.EMB'
tgt_path = '../../../vecmap/TRG_MAPPED-de-en-identical.EMB'

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

def csv_output(precision, recall):
    df = pd.read_csv(sys.argv[1])
    df["Precision"] = precision
    df["Recall"] = recall
    df.to_csv(sys.argv[1], index=False)


if __name__ == '__main__':

    translated_words = []
    reference = []
    with open(sys.argv[1], 'r') as file:
        matching = {}
        csv_reader = csv.reader(file)
        next(csv_reader)
        for row in csv_reader:
            for sentence in row:
                sentence = sentence.split('\t')
                res = [s.translate(str.maketrans('', '', string.punctuation)) for s in sentence]
                result_list = [elem.lower() for elem in res]
                for ref in result_list[0].split():
                    reference.append(ref)
                    matching = dict.fromkeys(reference, 0)

                for word in result_list[1].split():
                    translated_words.append(word)
                print(matching)
                # check given translations
                for sw in translated_words:
                    try:
                        # search top 10 translations
                        words = get_nn(sw, src_embeddings, src_id2word, tgt_embeddings, tgt_id2word, K=10)
                        matched = 0
                        for w in words:
                            if w in matching.keys():
                                matching[w] += 1
                        for key, value in matching.items():
                            if value >= 1:
                                matched += 1

                        precision = matched / len(translated_words)
                        print(matched)
                        print(precision)
                        recall = matched / len(reference)
                        print(recall)
                        csv_output(precision, recall)

                    except KeyError:
                        print("\"" + sw + "\"" + " not found in dictionary.")

    # try:
    #     # search if the given translation is in generated results
    #     #assert tgt_words[index] in words
    #     #print("\""+tgt_words[index]+"\""+ " is a matched word.", file=f)
    #
    # except AssertionError:
    #     # print("\""+tgt_words[index]+"\""+ " not in search results.", file=f)
    #     # print("Please check if the word is misspelled. List of Suggestion:", file=f)
    #     # print(list(spell.candidates(tgt_words[index])), file =f)

# print("Matched words" + " "+str(matched), file=f)
# print("Unmatched words" + " "+str(unmatched), file=f)
