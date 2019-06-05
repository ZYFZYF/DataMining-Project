# coding=utf8
import jieba
import codecs
import json
import pandas as pd
import gensim
from gensim.models.doc2vec import Doc2Vec, LabeledSentence
import logging
import numpy as np

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

TaggededDocument = gensim.models.doc2vec.TaggedDocument


def get_dataset():
    f = open('./data/reduce_titles.txt', 'r')
    x_train = []
    for i, line in enumerate(f.readlines()):
        words = line.split(' ')
        words[-1] = words[-1].strip()
        x_train.append(TaggededDocument(words, tags=[i]))
        if i % 100000 == 0:
            print('processed %s/%s' % (i, 20000000))
            if i == 5000000:
                break
    return x_train


def train_dm(x_train):
    model_dm = Doc2Vec(x_train, min_count=5, window=5, size=200, workers=8, iter=10)
    model_dm.save('./model/model_dm')


def train_dbow(x_train):
    model_dbow = Doc2Vec(x_train, min_count=5, window=5, size=200, workers=8, iter=10, dbow_words=1, dm=0)
    model_dbow.save('./model/model_dbow')


def train():
    x_train = get_dataset()
    train_dm(x_train)
    train_dbow(x_train)


def cos_dist(vec1, vec2):
    vector1 = np.array(vec1)
    vector2 = np.array(vec2)
    return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * (np.linalg.norm(vector2)))


model = Doc2Vec.load('./model/model_dm')


def get_sentence_vectors(sentence):
    return model.infer_vector(jieba.cut(sentence))


def test():
    sentences = ['用大蒜鉴别地沟油的方法,怎么鉴别地沟油', '一颗大蒜就能鉴别地沟油？别闹了！做到下面几点，让您远离地沟油',
                 '飞机就要起飞，一个男人在机舱口跪下！', '#吃秀##美好的一天从早餐开始#续集这会灯好多，上个一会黑一会白本来皮肤就黄',
                 '#营养美食季#糖尿病人到底能不能吃土豆？', '六月蚊子多多，千万别用蚊香！教你一招，家里屋子的蚊子全都死光',
                 '6月蚊子无法无天！蚊香无用不环保，教你一招妙计，蠢蚊子哭死了']
    for s1 in sentences:
        for s2 in sentences:
            print(s1)
            print(s2)
            print(cos_dist(get_sentence_vectors(s1), get_sentence_vectors(s2)))
    for s in sentences:
        print(s)
        for k, v in model.most_similar([model.infer_vector([s])]):
            print(k)
            print(v)


if __name__ == '__main__':
    # train()
    test()
    # print(cos_dist([1, 1], [2, 2]), cos_dist([1, 0], [0, 1]))
