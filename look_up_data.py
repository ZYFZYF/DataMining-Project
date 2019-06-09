# coding=utf8

import pandas as pd
import jieba.analyse
import seaborn as sbn
import matplotlib.pyplot as plt
from utils_common import TRAIN_FEATURES_NEW

allowPos = ['n', 'nr', 'nr1', 'nr2', 'ns', 'nrj', 'nt', 'nsf', 'nz', 't', 'v', 'vn', 'm', 'nl', 'ng']

if __name__ == '__main__':
    # df = pd.read_csv('./data/train.csv', encoding='utf8')
    # tf = df[df.label == u'agreed'][0:100]
    # for ind, row in tf.iterrows():
    #     print(' '.join([row['title1'], row['title2'], row['label']]))
    # tf = df[df.label == u'disagreed'][0:100]
    # for ind, row in tf.iterrows():
    #     print(' '.join([row['title1'], row['title2'], row['label']]))
    # df = pd.read_csv('./data/test.csv', encoding='utf8')
    # origin = set(df['id'].to_list())
    # tf = df.dropna(how='any', axis=0)
    # new = set(tf['id'].to_list())
    # print(len(df), len(tf), origin - new)
    # df = pd.read_csv('./data/train.csv')
    # df.dropna(how='any', inplace=True)
    # tf = df[df.label == u'disagreed']
    # print(len(tf))
    # sentence = ' '.join(tf['title2'].to_list())
    # print(sentence)
    # print(' '.join(jieba.analyse.extract_tags(sentence, topK=100, withWeight=False, allowPOS=['n','v'])))
    df = pd.read_csv('./data/train_categorical.csv')
    df = df[TRAIN_FEATURES_NEW + ['label']]
    df.rename(columns={'keywords_jaccard_distance': 'jaccard',
                       'title1_negative_count': 'title1',
                       'title2_negative_count': 'title2',
                       'keywords_meaning_distance_after_remove_negative_word': 'meaning'}, inplace=True)
    df.dropna(how='any', axis=0, inplace=True)
    sbn.pairplot(df, hue='label')
    plt.savefig('./pic/feature_select.png', dpi=300)
