from .util import get_stats
import re

s = 'AS##~@""DjifjASFJ7364'
delete_re = re.compile(r'[^a-zA-Z]+')
print(delete_re.sub('', s))

article_filename = 'data/croatian_war_of_independence.txt'

def sentence_words(sentence):
    for word in sentence.split(' '):
        word = delete_re.sub('', word)
        if word:
            yield word

def read_article_sentences(article_path):
    with open(article_path) as f:
        for line in f.readlines():
            for sentence in line.split('.'):
                sentence = list(sentence_words(sentence))
                if sentence:
                    yield sentence

if __name__ == "__main__":
    print(get_stats(len(x) for x in read_article_sentences(article_filename)))