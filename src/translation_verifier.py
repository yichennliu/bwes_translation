#!/usr/bin/python3
import sys

#insert your package path here
sys.path.append('/home/yibsimo/PycharmProjects/bwes_translation')
from src.streamlit.model import load_vec, get_nn
import csv
import string

src_emb = sys.argv[1]
tgt_emb = sys.argv[2]
output = sys.argv[4]


def main():
    with open(output, 'w') as op:
        writer = csv.writer(op, delimiter=',')
        header = ['Generics', 'Translation', 'Target Match Score', 'Source Match Score', 'Flag']
        writer.writerow(i for i in header)

        with open(sys.argv[3], 'r') as file:
            translation = []
            translated_words = []
            reference = []
            reference_words = []
            matching = {}
            csv_reader = csv.reader(file, delimiter='|', skipinitialspace=True)
            next(csv_reader)
            for row in csv_reader:
                for sentence in row:
                    sentence = sentence.split('\t')
                    translation.append(sentence[1])
                    reference.append(sentence[0])

                    inputs = []
                    inputs.append(sentence[0])
                    inputs.append(sentence[1])

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
                            for tup in words:
                                w = tup[0]
                                if w in matching.keys():
                                    matching[w] += 1

                        except KeyError:
                            with open("lexicon.txt", 'w') as lex:
                                lex.write(str("\"" + sw + "\"") + "\n")

                    for key, value in matching.items():
                        if value >= 1:
                            matched += 1

                    target_match = round(matched / len(translated_words), 2)
                    source_match = round(matched / len(reference_words), 2)

                    inputs.append(target_match)
                    inputs.append(source_match)

                    if target_match < 0.5:

                        inputs.append("1")

                    elif source_match < 0.5:

                        inputs.append("1")

                    else:

                        inputs.append("0")

                    writer.writerow(d for d in inputs)


if __name__ == '__main__':
    src_embeddings, src_id2word, src_word2id = load_vec(src_emb)
    tgt_embeddings, tgt_id2word, tgt_word2id = load_vec(tgt_emb)
    main()
