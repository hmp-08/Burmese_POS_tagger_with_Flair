# Burmese_POS_tagger_with_Flair
Burmese Part-of-Speech (POS) model Based on Flair NLP Library

# Introduction
Part-of-Speech (POS) tagging is the process of assigning grammatical categories to words in a sentence. It is crucial for various natural language processing (NLP) tasks such as parsing, information retrieval, named entity recognition, sentiment analysis, and machine translation. This project focuses on building a POS tagging model for the Burmese language using the Flair NLP library.

# Data
The annotated Burmese text data used in this project is sourced from Dr. Ye Kyaw Thu's GitHub repository, similar to the data used for the previous OpenNMT-based model. The data consists of word sequences with their corresponding POS tags. Each word is tokenized and annotated using the slash (/) delimiter.

# POS Tags
The model predicts 15 POS tags, including Abbreviation, Adjective, Adverb, Conjunction, Foreign Word, Interjection, Noun, Number, Particle, Post-positional Marker, Pronoun, Punctuation, Symbol, Text Number, and Verb.

# Custom Word Embeddings
To improve the model's performance for the Burmese language, a CustomEmbedding layer is implemented, allowing the use of pre-trained word2vec embeddings.

My pre-trained word2vec embedding model -gensim can be downloaded here -> 
[word2vec.model](https://drive.google.com/file/d/1Sa9TvWG0DMoGYdBe1ieUyw0tz_69C4PT/view?usp=sharing)

# How to Train
Training the Flair POS tagging model involves running the `pos_flair.ipynb` Jupyter Notebook or Python script that includes the necessary configurations, data loading, model training, and evaluation steps. Ensure that you have Flair and other required dependencies installed.
```python
# Sample training script

dataset_file = "mypos-ver.3.0-flair.txt"

# Define the columns in dataset
columns = {0: 'text', 1: 'pos'}

# Initialize the corpus
corpus = ColumnCorpus(data_folder='.', column_format=columns, train_file=dataset_file)
from flair.embeddings.base import EMBEDDING_CLASSES
model_path = "word2vec.model"
# Create a custom TokenEmbeddings object
custom_word_embeddings = CustomWordEmbeddings(model_path)

EMBEDDING_CLASSES.update({
    "custom_word_embeddings": custom_word_embeddings
})

label_type = 'pos'

# Create a label dictionary from the corpus
label_dict = corpus.make_label_dictionary(label_type=label_type)

# Create model
model = SequenceTagger(hidden_size=256,
                      embeddings=custom_word_embeddings,
                      tag_dictionary=label_dict,
                      tag_type=label_type)

# Create the trainer and train the model
trainer = ModelTrainer(model, corpus)
trainer.train('pos_tagger', learning_rate=0.1, mini_batch_size=32, max_epochs=10)
```
Flair NLP Library ---> https://github.com/flairNLP/flair

Dataset ---> [mypos-ver.3.0.txt](https://github.com/ye-kyaw-thu/myPOS/blob/master/corpus-ver-3.0/corpus/mypos-ver.3.0.txt)
