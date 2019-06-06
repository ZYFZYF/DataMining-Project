# coding=utf8
import jieba
import codecs
import json
import pandas as pd
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from train_doc_vectors import cos_dist, get_sentence_vectors
import logging
from collections import defaultdict
import seaborn
import math


def euclid_dist(vec1, vec2):
    res = 0
    for x, y in zip(vec1, vec2):
        res += (x-y) * (x-y)
    return math.sqrt(res)


def get_distance_function():
    # return cos_dist
    return euclid_dist


data = defaultdict(list)


def calc_distance(row):
    try:
        vec1 = get_sentence_vectors(row['title1'])
        vec2 = get_sentence_vectors(row['title2'])
        dist = get_distance_function()(vec1, vec2)
        data[row['label'].encode('utf8')].append(float(dist))
        return dist
    except Exception,e:
        print(e, row)
        return 0.0


def generate_distances():
    df = pd.read_csv('./data/train.csv', encoding='utf8')
    df.dropna(how='any', inplace=True, axis=0)
    df['distance'] = df.apply(calc_distance, axis=1)
    df.to_csv('./data/train_include_distance.csv', encoding='utf8', index=False)
    json.dump(data, open('./data/%s.json' % get_distance_function().__name__, 'w'))


def view():
    Tdata = json.load(open('./data/%s.json' % get_distance_function().__name__, 'r'))
    agree = Tdata['agreed']
    disagree = Tdata['disagreed']
    unrelated = Tdata['unrelated']
    seaborn.distplot(agree).get_figure().savefig('./pic/agree_distribute.png', dpi=300)
    seaborn.distplot(disagree).get_figure().savefig('./pic/disagree_distribute.png', dpi=300)
    seaborn.distplot(unrelated).get_figure().savefig('./pic/unrelated.png', dpi=300)
    seaborn.distplot(agree + disagree).get_figure().savefig('./pic/agree+disagree_distribute.png', dpi=300)


if __name__ == '__main__':
    generate_distances()
    view()