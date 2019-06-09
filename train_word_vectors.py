# coding=utf8
import jieba
import codecs
import json
import pandas as pd
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from train_doc_vectors import cos_dist
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def train():
    model = Word2Vec(LineSentence('./data/reduce_titles.txt'), sg=0, size=256, window=5, min_count=5, workers=8)
    model.save('./model/titles_cobw.word2vec')


model = Word2Vec.load('./model/titles_cbow.word2vec')


def test():
    print(model.accuracy)
    words = [u'中国', u'我', u'你', u'你好', u'辟谣', u'谣言', u'?', u'!', u'1']
    for word in words:
        print(word in model)
        print(model[word])
        if word in model:
            for ans in model.most_similar(word):
                print(ans[0].encode('utf8'))
                print(ans[1])


def get_similarity_of_two_words(w1, w2):
    return cos_dist(model[w1], model[w2])


def get_word_vector(word):
    return model[word]


if __name__ == '__main__':
    # train()
    test()
    print(get_word_vector(u'中国'))
    print(get_word_vector(u'得过'))
    print(get_similarity_of_two_words(u'塑料袋', u'塑料'))
    print(get_similarity_of_two_words(u'吃饭', u'塑料'))
    print(get_similarity_of_two_words(u'中国', u'美国'))



