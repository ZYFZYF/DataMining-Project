# coding=utf8
import jieba
import codecs
import json
import jieba.analyse
import pandas as pd
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from train_doc_vectors import cos_dist, get_sentence_vectors, jaccard_dist, allowPos
from train_word_vectors import get_word_vector
import logging
from collections import defaultdict
import seaborn
import math
import matplotlib.pyplot as plt
import numpy as np
from utils_logging import Logging

logging = Logging()


def euclid_dist(vec1, vec2):
    res = 0
    for x, y in zip(vec1, vec2):
        res += (x - y) * (x - y)
    return math.sqrt(res)


def get_distance_function():
    # return cos_dist
    # return euclid_dist
    return jaccard_dist


data = defaultdict(list)


def calc_distance(row):
    try:
        vec1 = get_sentence_vectors(row['title1'])
        vec2 = get_sentence_vectors(row['title2'])
        # dist = get_distance_function()(vec1, vec2)
        dist = get_distance_function()(row['title1'], row['title2'])
        data[row['label'].encode('utf8')].append(float(dist))
        return dist
    except Exception, e:
        print(e, row)
        return 0.0


def generate_distances():
    train = pd.read_csv('./data/train.csv', encoding='utf8')
    train = train[['id', 'title1', 'title2', 'label']]
    print(len(train))
    train.dropna(how='any', axis=0, inplace=True)
    print(len(train))
    test = pd.read_csv('./data/test.csv', encoding='utf8')
    test = test[['id', 'title1', 'title2']]
    print(len(test))
    test.dropna(how='any', axis=0, inplace=True)
    print(len(test))
    df = pd.concat([train, test], ignore_index=True)
    df = df[['id', 'title1', 'title2', 'label']]
    print(len(df))

    # df['distance'] = df.apply(calc_distance, axis=1)
    # df.to_csv('./data/train_include_distance.csv', encoding='utf8', index=False)
    # json.dump(data, open('./data/%s.json' % get_distance_function().__name__, 'w'))
    datas = df.values.tolist()
    logging.start('get jacard distance between two sentences')
    for ind, row in enumerate(datas):
        if ind % 5000 == 0:
            print("processed %s/%s" % (ind, len(datas)))
        try:
            dist = get_distance_function()(row[1], row[2])
        except Exception, e:
            print(e, row)
            dist = 0.0
        row.append(dist)
    logging.end()
    df = pd.DataFrame(data=datas, columns=['id', 'title1', 'title2', 'label', 'distance'])
    df.to_csv('./data/train_include_distance.csv', encoding='utf8', index=False)


def view():
    Tdata = json.load(open('./data/%s.json' % get_distance_function().__name__, 'r'))
    agree = Tdata['agreed']
    disagree = Tdata['disagreed']
    unrelated = Tdata['unrelated']
    seaborn.distplot(agree).get_figure().savefig('./pic/agree_distribute.png', dpi=300)
    plt.clf()
    seaborn.distplot(disagree).get_figure().savefig('./pic/disagree_distribute.png', dpi=300)
    plt.clf()
    seaborn.distplot(unrelated).get_figure().savefig('./pic/unrelated.png', dpi=300)
    plt.clf()
    seaborn.distplot(agree + disagree).get_figure().savefig('./pic/agree+disagree_distribute.png', dpi=300)
    plt.clf()


negative_words = [u'辟谣', u'假的', u'造谣', u'谣言', u'假消息', u'误读', u'编造', u'传谣', u'虚假', u'断章取义', u'纯属', u'误传', u'扯淡']


def calc_distance_remove_negative():
    pass


def generate_topic_distances():
    df = pd.read_csv('./data/train_include_distance.csv', encoding='utf8')
    datas = df.values.tolist()
    for ind, row in enumerate(datas):
        if ind % 5000 == 0:
            print('processed %s/%s' % (ind, len(datas)))
        title1_words = jieba.analyse.extract_tags(row[1], topK=20, withWeight=False, allowPOS=allowPos)
        title1_words_new = filter(lambda x: x not in negative_words, title1_words)
        row.append(len(title1_words) - len(title1_words_new))
        title2_words = jieba.analyse.extract_tags(row[2], topK=20, withWeight=False, allowPOS=allowPos)
        title2_words_new = filter(lambda x: x not in negative_words, title2_words)
        row.append(len(title2_words) - len(title2_words_new))
        word_vecs = []
        for word in title1_words_new:
            try:
                word_vec = get_word_vector(word)
                word_vecs.append(np.array(word_vec))
            except Exception, e:
                # print(word)
                pass
        if word_vecs:
            topic1 = reduce(lambda x, y: x + y, word_vecs) / len(word_vecs)
        else:
            print(row[1])
            print(' '.join(title1_words))
            print(' '.join(title1_words_new))
            topic1 = [0] * 256
        word_vecs = []
        for word in title2_words_new:
            try:
                word_vec = get_word_vector(word)
                word_vecs.append(np.array(word_vec))
            except Exception, e:
                # print(word)
                pass
        if word_vecs:
            topic2 = reduce(lambda x, y: x + y, word_vecs) / len(word_vecs)
        else:
            print(row[2])
            print(' '.join(title2_words))
            print(' '.join(title2_words_new))
            topic2 = [0] * 256
        distance = cos_dist(topic1, topic2)
        row.append(distance)
        if row[3] in []:
            print(row[3])
            print(row[1])
            print(' '.join(title1_words))
            print(' '.join(title1_words_new))
            print(row[2])
            print(' '.join(title2_words))
            print(' '.join(title2_words_new))
            print(distance)
    df = pd.DataFrame(data=datas, columns=['id', 'title1', 'title2', 'label', 'distance', 'title1_negative_count',
                                           'title2_negative_count', 'distance_after_remove_negative_word'])
    df.to_csv('./data/train_final.csv', encoding='utf8', index=False)


if __name__ == '__main__':
    # generate_distances()
    # view()
    generate_topic_distances()
