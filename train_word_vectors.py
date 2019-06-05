# coding=utf8
import jieba
import codecs
import json
import pandas as pd
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def train():
    model = Word2Vec(LineSentence('./data/reduce_titles.txt'), sg=0, size=256, window=5, min_count=5, workers=8)
    model.save('./model/titles_cobw.word2vec')


def test():
    model = Word2Vec.load('./model/titles_cbow.word2vec')
    print(model.accuracy)
    words = [u'中国', u'我', u'你', u'你好', u'辟谣', u'谣言']
    for word in words:
        print(word in model)
        if word in model:
            for ans in model.most_similar(word):
                print(ans[0].encode('utf8'))
                print(ans[1])


if __name__ == '__main__':
    # train()
    test()