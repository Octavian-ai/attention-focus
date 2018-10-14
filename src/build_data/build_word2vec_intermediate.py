import time
import numpy as np

from gensim.models import KeyedVectors

from .word2vec import encode_sequence
from .article2vec import read_article_sentences, article_filename
### Run this module to create the croation_war_of|_independence_vocab_*.npy files

def load_word2vec_model():
    # Download the word2vec embeddings from here https://code.google.com/archive/p/word2vec/
    model_path = "data/GoogleNews-vectors-negative300.bin"
    t_0 = time.time()
    model = KeyedVectors.load_word2vec_format(model_path, binary=True)
    print("Time to load word2vec", time.time() - t_0)
    return model


def get_vocab(article_path):
    model = load_word2vec_model()
    words = set()
    for sentence in read_article_sentences(article_path):
        words.update(x for x in sentence)
    print(len(words), "words in article")
    words_in_model = words.intersection(set(model.vocab))
    print(len(words_in_model), "words in article found in word2vec")
    vocab_vectors = np.zeros([len(words_in_model), model.vectors.shape[-1]], dtype=np.float32)
    vocab_words = np.empty([len(words_in_model)], dtype=np.object)
    for i, embedding in enumerate(encode_sequence(words_in_model, model)):
        vocab_words[i] = embedding[0]
        vocab_vectors[i] = embedding[1]
    return vocab_words, vocab_vectors


def save_article_vocab():
    words, vectors = get_vocab(article_filename)
    np.save(article_filename.split('.')[0] + '_vocab_words.npy', words, allow_pickle=True)
    np.save(article_filename.split('.')[0] + '_vocab_vectors.npy', vectors, allow_pickle=False)

if __name__ == "__main__":
    save_article_vocab()