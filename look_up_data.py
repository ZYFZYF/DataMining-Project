# coding=utf8

import pandas as pd


if __name__ == '__main__':
    # df = pd.read_csv('./data/train.csv', encoding='utf8')
    # tf = df[df.label == u'agreed'][0:100]
    # for ind, row in tf.iterrows():
    #     print(' '.join([row['title1'], row['title2'], row['label']]))
    # tf = df[df.label == u'disagreed'][0:100]
    # for ind, row in tf.iterrows():
    #     print(' '.join([row['title1'], row['title2'], row['label']]))
    df = pd.read_csv('./data/test.csv', encoding='utf8')
    origin = set(df['id'].to_list())
    tf = df.dropna(how='any', axis=0)
    new = set(tf['id'].to_list())
    print(len(df), len(tf), origin - new)
