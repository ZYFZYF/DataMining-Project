# coding=utf8
import jieba
import codecs
import json
import jieba.analyse
import pandas as pd
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from train_doc_vectors import cos_dist, get_sentence_vectors, jaccard_dist, allowPos, jaccard_dist_between_list, word2vec_dist
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
    # return word2vec_dist


data = defaultdict(list)


def calc_distance(row):
    try:
        # dist = get_distance_function()(vec1, vec2)
        dist = get_distance_function()(row['title1'], row['title2'])
        return dist
    except Exception, e:
        print(e, row)
        return 0.0


def data_clean():
    train = pd.read_csv('./data/train.csv', encoding='utf8')
    train = train[['id', 'title1', 'title2', 'label']]
    logging.info('origin have %s rows in train_data' % len(train))
    train.dropna(how='any', axis=0, inplace=True)
    logging.info('now have %s rows in train_data after remove NULL rows' % len(train))
    test = pd.read_csv('./data/test.csv', encoding='utf8')
    test = test[['id', 'title1', 'title2']]
    logging.info('origin have %s rows in test_data' % len(test))
    test.dropna(how='any', axis=0, inplace=True)
    logging.info('now have %s rows in train_data after remove NULL rows' % len(test))
    df = pd.concat([train, test], ignore_index=True)
    df = df[['id', 'title1', 'title2', 'label']]
    logging.info('totally we have %s rows in data' % len(df))
    df.to_csv('./data/train_after_clean.csv', encoding='utf8', index=False)


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


negative_words = [u'辟谣', u'假的', u'造谣', u'谣言', u'假消息', u'误读', u'编造', u'传谣', u'虚假', u'断章取义', u'纯属',
                  u'扯淡', u'错了', u'传谣', u'否认', u'传言', u'假新闻', u'诽谤', u'不实', u'假的', u'谣传', u'澄清', u'不实',
                  u'消息不实']


def calc_distance_remove_negative():
    pass


def extract_features():
    df = pd.read_csv('./data/train_after_clean.csv', encoding='utf8')
    datas = df.values.tolist()
    drop_cnt = 0
    for ind, row in enumerate(datas):
        if ind % 5000 == 0:
            print('processed %s/%s' % (ind, len(datas)))
            # break
        error = 0
        # 两个句子关键词的jaccard距离
        try:
            dist = get_distance_function()(row[1], row[2])
        except Exception, e:
            print(e, row)
            dist = 0.0
            error += 1
        row.append(dist)

        title1_words = jieba.analyse.extract_tags(row[1], topK=20, withWeight=False, allowPOS=allowPos)
        title1_words_new = filter(lambda x: x not in negative_words, title1_words)
        if len(title1_words) - len(title1_words_new):
            row.append(1)
        else:
            row.append(0)
        title2_words = jieba.analyse.extract_tags(row[2], topK=20, withWeight=False, allowPOS=allowPos)
        title2_words_new = filter(lambda x: x not in negative_words, title2_words)
        if len(title2_words) - len(title2_words_new):
            row.append(1)
        else:
            row.append(0)

        # 算去除负面词之后的语义距离
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
            print(' '.join(title1_words_new))
            topic1 = [0] * 256
            error += 1
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
            print(' '.join(title2_words_new))
            topic2 = [0] * 256
            error += 1
        distance = cos_dist(topic1, topic2)
        row.append(distance)

        # 算去除之前的语义距离
        word_vecs = []
        for word in title1_words:
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
            topic1 = [0] * 256
            error += 1
        word_vecs = []
        for word in title2_words:
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
            topic2 = [0] * 256
            error += 1
        distance = cos_dist(topic1, topic2)
        row.append(distance)

        # 加一个去除之后的jaccard距离
        try:
            row.append(jaccard_dist_between_list(title1_words_new, title2_words_new))
        except Exception,e:
            row.append(0.0)
            error += 1
        row.append(' '.join(title1_words))
        row.append(' '.join(title2_words))

        if row[3] in [u'agreed', u'disagreed', u'unrelated'] and error > 0:
            row.pop()
            row.pop()
            row.pop()
            drop_cnt += 1
            print('dropped id=%s, totally dropped %s/%s' % (row[0], drop_cnt, ind))
    df = pd.DataFrame(data=datas,
                      columns=['id', 'title1', 'title2', 'label', 'keywords_jaccard_distance', 'title1_negative_count',
                               'title2_negative_count', 'keywords_meaning_distance_after_remove_negative_word',
                               'keywords_meaning_distance', 'keywords_jaccard_distance_after_remove_negative_word',
                               'title1_cut_words', 'title2_cut_words'])
    df.to_csv('./data/train_categorical_final.csv', encoding='utf8', index=False)


if __name__ == '__main__':
    data_clean()
    # view()
    extract_features()
