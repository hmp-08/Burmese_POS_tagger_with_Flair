# Burmese_POS_tagger_with_Flair
Burmese Part-of-Speech (POS) model Based on Flair NLP Library

# Introduction
Part-of-Speech (POS) tagging is the process of assigning grammatical categories to words in a sentence. It is crucial for various natural language processing (NLP) tasks such as parsing, information retrieval, named entity recognition, sentiment analysis, and machine translation. This project focuses on building a POS tagging model for the Burmese language using the Flair NLP library.

# Data
The annotated Burmese text data used in this project is sourced from Dr. Ye Kyaw Thu's GitHub repository. The data consists of word sequences with their corresponding POS tags. Each word is tokenized and annotated using the slash (/) delimiter.

# POS Tags
The model predicts 15 POS tags, including Abbreviation, Adjective, Adverb, Conjunction, Foreign Word, Interjection, Noun, Number, Particle, Post-positional Marker, Pronoun, Punctuation, Symbol, Text Number, and Verb.

## 1.Custom Word Embeddings (word2vec)
To improve the model's performance for the Burmese language, a CustomEmbedding layer is implemented, allowing the use of pre-trained word2vec embeddings.

My pre-trained word2vec embedding model -gensim can be downloaded here -> 
[word2vec.model](https://drive.google.com/file/d/1Sa9TvWG0DMoGYdBe1ieUyw0tz_69C4PT/view?usp=sharing)

![](https://github.com/hmp-08/Burmese_POS_tagger_with_Flair/blob/main/Screenshot%20from%202023-12-20%2013-48-57.png)

## How to Train
Training the Flair POS tagging model involves running the `flair_pos_word2vec_embedding.ipynb` Jupyter Notebook or Python script that includes the necessary configurations, data loading, model training, and evaluation steps. Ensure that you have Flair and other required dependencies installed.
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
## 2. Flair Embeddings
My pre-trained Flair embedding models (char level) forward and backward can be downloaded here ->
[forward](https://drive.google.com/file/d/18fLAETu7aisMcVQR5VWmh3ZbOFTyLbAg/view?usp=sharing)
[backward](https://drive.google.com/file/d/1vKbbMyuUcijKYxuWQ4bi6rTFnrJL7roO/view?usp=sharing)

## How to Train
Training the Flair POS tagging model involves running the `flair_pos_flairembeddings.ipynb` Jupyter Notebook or Python script that includes the necessary configurations, data loading, model training, and evaluation steps. 

```python
# Sample training script

dataset_file = "mypos-ver.3.0-flair.txt"

# Define the columns in dataset
columns = {0: 'text', 1: 'pos'}

# Initialize the corpus
corpus = ColumnCorpus(data_folder='.', column_format=columns, train_file=dataset_file)

from flair.embeddings import FlairEmbeddings, StackedEmbeddings, WordEmbeddings
flair_forward_embedding = FlairEmbeddings('best-lm-fw.pt')
flair_backward_embedding = FlairEmbeddings('best-lm-bw.pt')

# Stack the embeddings
embedding_types = [

    FlairEmbeddings('best-lm-fw.pt'),
    FlairEmbeddings('best-lm-bw.pt')
]

embeddings = StackedEmbeddings(embeddings=embedding_types)

label_type = 'pos'

# Create a label dictionary from the corpus
label_dict = corpus.make_label_dictionary(label_type=label_type, add_unk=True)

# Create model
model = SequenceTagger(hidden_size=256,
                      embeddings=embeddings,
                      tag_dictionary=label_dict,
                      tag_type=label_type)

# Create the trainer and train the model
trainer = ModelTrainer(model, corpus)
trainer.train('pos_tagger', learning_rate=0.1, mini_batch_size=32, max_epochs=10)
```

Flair NLP Library ---> https://github.com/flairNLP/flair

Dataset ---> [mypos-ver.3.0.txt](https://github.com/ye-kyaw-thu/myPOS/blob/master/corpus-ver-3.0/corpus/mypos-ver.3.0.txt)
