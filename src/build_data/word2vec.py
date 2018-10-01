import time
import re
import random
import numpy as np

from gensim.models import KeyedVectors

from .article2vec import read_article_sentences, article_filename

s = 'AS##~@""DjifjASFJ7364'
delete_re = re.compile(r'[^a-zA-Z]+')
print(delete_re.sub('', s))


def load_word2vec_model():
    # Download the word2vec embeddings from here https://code.google.com/archive/p/word2vec/
    model_path = "data/GoogleNews-vectors-negative300.bin"
    t_0 = time.time()
    model = KeyedVectors.load_word2vec_format(model_path, binary=True)
    print("Time to load word2vec", time.time() - t_0)
    return model


def encode_sequence(seq, model):
    for word in seq:
        if word in model:
            encoding = model[word]
            yield word, encoding


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


def query_in_list(list, query):
    assert len(query.shape) == 1
    return np.equal(list, query).all(1).any()


SENTENCE_LENGTH = 77  # max in article was 77


def get_sentences():
    words = np.load(article_filename.split('.')[0] + '_vocab_words.npy')
    vectors = np.load(article_filename.split('.')[0] + '_vocab_vectors.npy')
    vocab = VocabFromNumpyArrays(words, vectors)

    N = sum(1 for _ in read_article_sentences(article_filename))
    sentences = np.zeros([N, SENTENCE_LENGTH, vectors.shape[-1]])
    for i, sentence in enumerate(read_article_sentences(article_filename)):
        s = np.zeros([SENTENCE_LENGTH, vectors.shape[-1]], dtype=np.float32)
        for j, encoding in enumerate(encode_sequence(sentence, vocab)):
            if j >= SENTENCE_LENGTH:
                break
            s[j] = encoding[1]
        sentences[i] = s
    return vectors, sentences


class VocabFromNumpyArrays(object):
    def __init__(self, words, vectors):
        self.words = words
        self.vectors = vectors
        assert vectors[0].shape == (300,)

    def __contains__(self, word):
        return np.equal(self.words, word).any()

    def __getitem__(self, word):
        i = np.where(self.words == word)[0]
        return self.vectors[np.random.choice(i)]


#save_article_vocab()

t_0 = time.time()
print("loading sentences")
WORDS, SENTENCES = get_sentences()
print("loading sentences", time.time() - t_0)

def random_croatia_vector():
    return WORDS[np.random.choice(len(WORDS), 1, replace=False)].flatten()


def random_croatia_set(sz, query):
    assert sz == SENTENCE_LENGTH
    if bool(random.getrandbits(1)):
        # Choose a sentence that contains the word
        s = SENTENCES[np.random.choice(np.where(np.equal(SENTENCES, query).all(-1).any(-1))[0])]
    else:
        # Choose a sentence that does not contain the word
        s = SENTENCES[np.random.choice(np.where(np.not_equal(SENTENCES, query).all(-1).all(-1))[0])]
    # Do shuffle the arrays to add a bit more variety to the task
    s = np.copy(s)
    np.random.shuffle(s)
    return s