import time
import random
import numpy as np

from .util import once
from .article2vec import read_article_sentences, article_filename

def encode_sequence(seq, model):
    for word in seq:
        if word in model:
            encoding = model[word]
            yield word, encoding


def query_in_list(list, query):
    assert len(query.shape) == 1
    return np.equal(list, query).all(1).any()


SENTENCE_LENGTH = 77

@once
def get_sentences():
    t_0 = time.time()
    print("loading sentences...")
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


    print("loaded sentences", time.time() - t_0)
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


def random_croatia_vector():
    WORDS, SENTENCES = get_sentences()
    return WORDS[np.random.choice(len(WORDS), 1, replace=False)].flatten()


def random_croatia_set(sz, query):
    WORDS, SENTENCES = get_sentences()
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
