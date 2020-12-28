#!/usr/bin/python3
# -*- coding: utf-8 -*-

import re
from collections import Counter

from mosestokenizer import *
from nltk.corpus import stopwords
import sys

def tok_and_lower(infile, output):
    tokenize = MosesTokenizer('en')
    for line in infile:
        tokenize(line)
        line = line.strip().split()
        lower = [word.lower() for word in line]
        output.write(" ".join(str(w) for w in lower) + "\n")


# remove special characters from the datasets
def remove_noise(infile, outfile):
    with open(outfile, 'w') as f:
        for line in open(infile):
            extract_words = line.split()
            # print(extract_words)
            for word in extract_words:
                filtered_word = re.sub('^&apos;|^&quot;', ' ', word)
                f.write((filtered_word + " "))


# remove digits, punctuation, and stopwords for gold standard lexicon creation
def remove_digit_punct_stopword(infile, outfile, lang):
    set_stopword = set(stopwords.words(lang))
    with open(outfile, 'w', encoding='UTF8') as f:
        for line in open(infile, 'r', encoding='UTF8'):
            # split into words
            extract_words = line.split()
            for word in extract_words:
                filtered_word = re.sub('[^a-zöäüßA-ZÖÄÜ\s]', '', word)
                if not filtered_word in set_stopword:
                    if not filtered_word.isdigit():
                        f.writelines(filtered_word + "\n")


# build goldstandard lexicons with most frequent words
def extract_most_frequent(infile, outfile):
    with open(outfile, 'w', encoding='UTF8') as file:
        with open(infile, "r", encoding='UTF8') as input:
            count = Counter(line for line in input)
            for word in count.most_common(2000):
                file.writelines(word[0] + "\n")


def calculate_words(infile):
    count = 0
    with open(infile, "r") as inp:
        for line in inp:
            count += 1

    print("Gold standard lexicon has " + count + " words.")


def skip_empty_line(infile, outfile):
    with open(outfile, 'w', encoding='UTF8') as f:
        with open(infile, 'r', encoding='UTF8') as file:
            for line in file:
                line = line.strip()
                if not line:  # line is blank
                    continue
                else:
                    f.writelines(line + "\n")


if __name__ == '__main__':
    tok_and_lower(sys.argv[1], sys.argv[2])
