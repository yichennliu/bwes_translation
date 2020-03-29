# Improving Translation Quality of Survey Questions with Bilingual Word Embeddings

Translation has been one of the most important tasks that strengthens the communication and enables the exchange of all kinds of knowledge between people.
This task is not only performed by human translators, but also by machine translation systems. Creating machine translation systems needs a lot of parallel corpora, 
which is expensive and might not be available for some languages and specific domains.
In this Bachelor thesis, unsupervised bilingual word embeddings are created by exploiting monolingual corpora of source language and target language. 
This approach enables to mine translations under circumstances where there is a limited amount of parallel corpora in different languages as well as in specific domains. The performance of the approach is evaluated with different word translation tasks on different domains that simulate the translation and translation verification tasks in the Survey of Health, Ageing and Retirement in Europe (SHARE) project, which is a research project that aims to improve social policies through studying the living status of aging population in Europe.


### Prerequisites

* Python 3.5 
* Dependencies in requirements.txt

```
pip install -r requirements.txt
```

### Usage

To run translation verification, bilingual word embeddings need to be created. For tasks in SHARE, We use [Vecmap](https://github.com/artetxem/vecmap) to train them. The trained embeddings
are available in the SHARE drive.

```
python3 SRC_MAPPED.EMB TRG_MAPPED.EMB csv_file
```

For SHARE translation verification, German word embeddings (as SRC_MAPPED.EMB) are projected into English word embeddings (as TRG_MAPPED.EMB). For example:

```
python3 SRC_MAPPED_de-en.EMB TRG_MAPPED_de-en.EMB demo_csv_file

```

The precision and recall matching scores will be printed out in the terminal. Scores that are lower than 0.50 will be marked in red, implying the verifiers that they should recheck the human translation.



