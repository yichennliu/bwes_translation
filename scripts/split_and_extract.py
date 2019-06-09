import random
import shutil
from math import floor

en_gen_data = ["../../Dataset/SH_gen_en_tok_lc.txt", "../../Dataset/news_tok_lcased.en",
               "../../Dataset/europarl-v9-tok-lc.en", "../../Dataset/tok_lcased.nc.en"]
de_gen_data = ["../../Dataset/SH_gen_de_tok_lc.txt", "../../Dataset/news_tok_lcased.de",
               "../../Dataset/europarl-v9-tok-lc.de", "../../Dataset/tok_lcased.nc.de"]
fr_gen_data = ["../../Dataset/SH_gen_fr_tok_lc.txt", "../../Dataset/news_tok_lcased.fr",
               "../../Dataset/europarl-v7-tok-lc.fr", "../../Dataset/tok_lcased.nc.fr"]

en_med_data = ["../../Dataset/SH_med_en_tok_lc.txt", "../../Dataset/medical_en_tok_lc.txt"]
de_med_data = ["../../Dataset/SH_med_de_tok_lc.txt", "../../Dataset/medical_de_tok_lc.txt"]
fr_med_data = ["../../Dataset/SH_med_fr_tok_lc.txt", "../../Dataset/medical_fr_tok_lc.txt"]


# split the dataset into 70% train set and 30% test set
def split_dataset(data):
    random.shuffle(data)
    split_id = floor(len(data) * 0.7)
    training = data[:split_id]
    testing = data[split_id:]
    return training, testing


# build the dataset
def extract_files(final_data, shuffle_op):
    with open(shuffle_op, 'wb') as shuf:
        for f in final_data:
            with open(f, 'rb') as input:
                shutil.copyfileobj(input, shuf)


if __name__ == '__main__':
    en_gen_train, en_gen_test = split_dataset(en_gen_data)
    de_gen_train, de_gen_test = split_dataset(de_gen_data)
    fr_gen_train, fr_gen_test = split_dataset(fr_gen_data)
    en_med_train, en_med_test = split_dataset(en_med_data)
    de_med_train, de_med_test = split_dataset(de_med_data)
    fr_med_train, fr_med_test = split_dataset(fr_med_data)

    extract_files(en_gen_train, "../../shuffle-gen-en-train")
    extract_files(en_gen_test, "../../shuffle-gen-en-test")
    extract_files(de_gen_train, "../../shuffle-gen-de-train")
    extract_files(de_gen_test, "../../shuffle-gen-de-test")
    extract_files(fr_gen_train, "../../shuffle-gen-fr-train")
    extract_files(fr_gen_test, "../../shuffle-gen-fr-test")
    extract_files(en_med_train, "../../shuffle-med-en-train")
    extract_files(en_med_test, "../../shuffle-med-en-test")
    extract_files(de_med_train, "../../shuffle-med-de-train")
    extract_files(de_med_test, "../../shuffle-med-de-test")
    extract_files(fr_med_train, "../../shuffle-med-fr-train")
    extract_files(fr_med_test, "../../shuffle-med-fr-test")
