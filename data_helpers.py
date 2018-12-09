import numpy as np
import re
import itertools
from collections import Counter
import pandas as pd


def normalize(data):
    data = data.values
    print(data[0])


def clean_str(string):
    """
    Tokenization/string cleaning for datasets.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels():
    """
    Loads polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open("./data/rt-polarity.pos", "r", encoding='latin-1').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open("./data/rt-polarity.neg", "r", encoding='latin-1').readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    x_text = [s.split(" ") for s in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def pad_sentences(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + padding_word * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    print("Build vocabulary")
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    print("Mapping from index to word")
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    vocabulary_inv = list(sorted(vocabulary_inv))
    # Mapping from word to index
    print("Mapping from word to index")
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentences and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]


def load_data():
    """
    Loads and preprocessed data for the dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    #sentences, labels = load_data_and_labels()
    print("accessing data/compiled_data.csv")
    sentences, labels = load_dataset("final_data/compiled_data.csv")
    print("padding sentences")
    sentences_padded = pad_sentences(sentences)
    print("building vocabulary")
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    print("building input data")
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv]

#our load data and labels
def load_dataset(csv_path):

    print("*******************loading dataset*******************")

    data = pd.read_csv(csv_path, encoding = "ISO-8859-1");
    num_data_points = len(data.index);

    data = normalize(data)

    # train_tweets = data['Tweet content'].values[0:int(num_data_points * TRAIN_SPLIT)]
    # val_tweets = data['Tweet content'].values[int(num_data_points * TRAIN_SPLIT): int(num_data_points * TRAIN_SPLIT)+ int(num_data_points * VALIDATION_SPLIT)]
    # test_tweets = data['Tweet content'].values[int(num_data_points * TRAIN_SPLIT)+ int(num_data_points * VALIDATION_SPLIT):]

    #splits data into train, val, and tests
    # train_labels = data['increase'].values[0:int(num_data_points * TRAIN_SPLIT)]
    # val_labels = data['increase'].values[int(num_data_points * TRAIN_SPLIT): int(num_data_points * TRAIN_SPLIT)+ int(num_data_points * VALIDATION_SPLIT)]
    # test_labels = data['increase'].values[int(num_data_points * TRAIN_SPLIT)+ int(num_data_points * VALIDATION_SPLIT):]
    # return train_tweets, val_tweets, test_tweets, train_labels, val_labels, test_labels

    tweets = data['Tweet content'].values
    labels = data['increase'].values
    return [tweets, labels]
