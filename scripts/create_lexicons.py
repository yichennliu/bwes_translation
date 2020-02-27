#!/usr/bin/python3
# -*- coding: utf-8 -*-

import re
from collections import Counter
from nltk.corpus import stopwords


# remove digits, punctuation, and stopwords for lexicon creation
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

# remove other noises from the datasets
def remove_noise(infile, outfile):
    with open(outfile, 'w') as f:
        for line in open(infile):
            extract_words = line.split()
            #print(extract_words)
            for word in extract_words:
                filtered_word =  re.sub('^&apos;|^&quot;',' ', word)
                f.write((filtered_word + " "))


# build goldstandard lexicons
def extract_most_frequent(infile, outfile):
    with open(outfile, 'w', encoding= 'UTF8') as file:
        with open(infile, "r", encoding= 'UTF8') as input:
            count = Counter(line for line in input)
            for word in count.most_common(5000):
                file.writelines(word[0] + "\n")


# calculate words in gold-standard lexicons
def calculate_words(infile):
    count = 0
    with open(infile, "r") as inp:
        for line in inp:
            count += 1

    print(count)

def split_text(file):
    with open(file, 'r') as f:
        for line in f.read():
            line.split()


def skip_empty_line(infile, outfile):
    with open(outfile, 'w', encoding= 'UTF8') as f:
        with open(infile, 'r', encoding='UTF8') as file:
            for line in file:
                line = line.strip()
                if not line:  # line is blank
                    continue
                else:
                    f.writelines(line+"\n")


def create_dictionary(file1, file2, endfile):
    with open(file1,'r', encoding="UTF8") as f1:
        with open(file2, 'r', encoding="UTF8") as f2:
            with open(endfile, 'w') as result:
                # Read first file
                f1lines = f1.readlines()
                # Read second file
                f2lines = f2.readlines()
                # Combine content of both lists  and Write to third file
                for line1, line2 in zip(f1lines, f2lines):
                    result.write("{} {}\n".format(line1.rstrip(), line2.rstrip()))

def remove_line(infile, outfile):
   with open(outfile, 'w') as f:
        with open(infile, 'r') as file:
            for line in file:
                line= line.strip().split()
                if len(line) == 2:
                    f.write(" ".join(str(word) for word in line)+"\n")




if __name__ == '__main__':
    # remove_digit_punct_stopword("../../../shuffle-gen-en-train", "../../../removed_noise_gen-en", 'english')
    # remove_digit_punct_stopword("../../../shuffle-gen-de-train", "../../../removed_noise_gen-de", 'german')
    # extract_most_frequent("../../../removed_noise_gen-en", "../../../en_freq")
    # extract_most_frequent("../../../removed_noise_gen-de", "../../../de_freq")
    # skip_empty_line("../../../Seed_de_en_translated","../../../seed_de-en-ok")
    # skip_empty_line("../../../Seed_en_de_translated", "../../../seed_en-de-ok")
    # create_dictionary("../../../seed_en-de","../../../seed_en-de-ok", "../../../seed_EN-DE")
    # create_dictionary("../../../seed_de-en", "../../../seed_de-en-ok", "../../../seed_DE-EN")
    remove_line("../../../seed_DE-EN", "../../../finished_DE-EN")