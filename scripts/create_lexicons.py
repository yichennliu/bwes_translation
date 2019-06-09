import re
from collections import Counter

from nltk.corpus import stopwords


# remove digits, punctuation, and stopwords for lexicon creation
def remove_digit_punct_stopword(infile, outfile, lang):
    set_stopword = set(stopwords.words(lang))
    with open(outfile, 'w') as f:
        for line in open(infile):
            # split into words
            extract_words = line.split()
            for word in extract_words:
                filtered_word = re.sub('[^a-zA-Z \s\d+]', '', word)
                if not filtered_word in set_stopword:
                    f.write(filtered_word + "\n")


# build goldstandard lexicons
def extract_most_frequent(infile, outfile):
    with open(outfile, 'w') as file:
        with open(infile, "r") as input:
            count = Counter(line for line in input)
            for word in count.most_common(500):
                file.write(word[0] + "\n")


# calculate words in gold-standard lexicons
def calculate_words(infile):
    count = 0
    with open(infile, "r") as inp:
        for line in inp:
            count += 1

    print(count)


if __name__ == '__main__':
    remove_digit_punct_stopword("../../shuffle-en-train-med", "../../removed_med-en", "english")
    extract_most_frequent("../../removed_med-en", "../../frequent_med-en")
