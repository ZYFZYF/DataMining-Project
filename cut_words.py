# coding=utf8
import jieba
import codecs
import json
import pandas as pd
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
config = json.load(open('config.json', 'r'))

if __name__ == '__main__':
    chunk_size = 100000
    f = open('./data/reduce_titles.txt', 'w')
    cnt = 0
    for df in pd.read_csv('./data/toutiao_titles.csv', encoding='utf8', chunksize=chunk_size):
        df.dropna(how='any', axis=0, inplace=True)
        df['title'] = df['title'].astype('object').apply(lambda x: ' '.join(jieba.cut(x)))
        for line in df['title'].tolist():
            f.write(line.encode('utf8') + '\n')
        cnt += chunk_size
        print('processed %s/%s' % (cnt, 20000000))
    f.close()
