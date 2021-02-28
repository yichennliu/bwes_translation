# Improving Translation Quality of Survey Questions with Bilingual Word Embeddings

Translation has been one of the most important tasks that strengthens the communication and enables the exchange of all kinds of knowledge between people.
This task is not only performed by human translators, but also by machine translation systems. Creating machine translation systems needs a lot of parallel corpora, 
which is expensive and might not be available for some languages and specific domains.
In this translation verification project, unsupervised bilingual word embeddings are created by exploiting monolingual corpora of source language and target language.
This approach enables to mine translations under circumstances where there is a limited amount of parallel corpora in different languages as well as in specific domains. The performance of the approach is evaluated with different word translation tasks on different domains that simulate the translation and translation verification tasks in the Survey of Health, Ageing and Retirement in Europe ([SHARE](http://www.share-project.org/home0.html)) project, which is a research project that aims to improve social policies through studying the living status of aging population in Europe.


## Prerequisites

* Python 3.5 or later
* Dependencies in requirements.txt

```
pip install -r requirements.txt
```

## Usage

Before running translation verification, bilingual word embeddings need to be created. For tasks in SHARE, we are using [Vecmap](https://github.com/artetxem/vecmap) to train them.


### Streamlit

We use Streamlit to display the verification process and visualize our current trained model. Currently, we are working on German to English translation verification. For further information, run the following command:

```
streamlit run app.py
```

### Customization

Import your own trained bilingual word embeddings with language pairs in your favor and run the following command:

```
python3 translation_verifier.py SRC_MAPPED.EMB TRG_MAPPED.EMB file_to_be_checked.csv output_results.csv
```

For SHARE translation verification, German word embeddings (as SRC_MAPPED.EMB) are projected onto English word embeddings (as TRG_MAPPED.EMB). For example:

```
python3 translation_verifier.py SRC_MAPPED_de-en.EMB TRG_MAPPED_de-en.EMB demo.csv results.csv
```

The results are listed in a csv file. Target match scores that are lower than 0.50 will be flagged (noted as 1), implying human verifiers that they should recheck their translation. For more detail about how the csv files are created, please check the demo.csv and results.csv in the data directory.

## Presentations

We have presented this approach in [BigSurv20](https://www.bigsurv20.org/) and seminars in [Max Planck Institute for Social Law and Social Policy](https://www.mpg.de/149954/sozialrecht). Presentation slides and documentation could be found in the data directory.


## Acknowledgement

This project was developed and tested as part of H2020 EU Project [SSHOC](https://sshopencloud.eu/)
![Alt text](data/img/sshoc_logo.png)
![Alt text](data/img/sshoc_eu_tag.png)


