#!/usr/bin/python3
import os
import sys

sys.path.append('/home/yibsimo/PycharmProjects/bwes_translation')
from src.streamlit.model import *

import string
import base64
import pandas as pd

st.title("Word Embeddings Visaulization")
st.markdown("Visualizing Bilingual Word Embeddings")

src_embeddings, src_id2word, src_word2id = load_vec(
    r"/home/yibsimo/PycharmProjects/bwes_translation/SRC_MAPPED_de-en.EMB")
tgt_embeddings, tgt_id2word, tgt_word2id = load_vec(
    r"/home/yibsimo/PycharmProjects/bwes_translation/TRG_MAPPED_de-en.EMB")


def insert(df, row):
    insert_loc = df.index.max()

    if pd.isna(insert_loc):
        df.loc[0] = row
    else:
        df.loc[insert_loc + 1] = row


def create_download_link(data):
    """
    in:  dataframe
    out: href string
    """
    dataframe = data.to_csv(index=False)
    b64 = base64.b64encode(dataframe.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as &lt;some_name&gt;.csv)'
    return href


@st.cache(persist=True)
def verify(infile, outfile):
    data = pd.read_csv(infile)
    translation = []
    translated_words = []
    reference = []
    reference_words = []
    matching = {}

    outfile = pd.DataFrame(columns=['Generics', 'Translation', 'Target Match Score', 'Source Match Score', 'Flag'])

    for idx, value in data.iterrows():

        generics = value.str.strip().loc['Source']
        given_translation = value.str.strip().loc['Translated']
        reference.append(generics)
        translation.append(given_translation)

        inputs = []
        inputs.append(generics)
        inputs.append(given_translation)

        res = [ip.translate(str.maketrans('', '', string.punctuation)) for ip in inputs]
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
                "Word not found."

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

        insert(outfile, inputs)

    href = create_download_link(outfile)

    return href


def plot_similar_word(src_words, src_word2id, src_emb, tgt_words, tgt_word2id, tgt_emb, pca):
    Y = []
    word_labels = []
    for sw in src_words:
        Y.append(src_emb[src_word2id[sw]])
        word_labels.append(sw)
    for tw in tgt_words:
        Y.append(tgt_emb[tgt_word2id[tw]])
        word_labels.append(tw)

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

    st.pyplot()


upload_file = st.sidebar.file_uploader("Upload File", type="csv")
show_file = st.sidebar.empty()

if not upload_file:
    show_file.info("Please upload the csv file")

if st.sidebar.button("Verify", key='verify'):
    href = verify(upload_file, "result.csv")
    st.markdown(href, unsafe_allow_html=True)

pca = PCA(n_components=2, whiten=True)  # TSNE(n_components=2, n_iter=3000, verbose=2)
pca.fit(np.vstack([src_embeddings, tgt_embeddings]))
# print('Variance explained: %.2f' % pca.explained_variance_ratio_.sum())


st.sidebar.subheader("Type German translation you want to check: ")
user_input_source = st.sidebar.text_input("German Translation Input", "König")
user_input_target = st.sidebar.text_input("English Translation", "King")

src_words = [user_input_source.lower()]
tgt_words = [user_input_target.lower()]

if st.sidebar.button("Visualize", key='visualize'):
    plot_similar_word(src_words, src_word2id, src_embeddings, tgt_words, tgt_word2id, tgt_embeddings, pca)
