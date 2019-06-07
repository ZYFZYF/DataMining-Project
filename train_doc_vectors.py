# coding=utf8
import jieba
import jieba.analyse
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
    stopwords = get_stopwords()
    for i, line in enumerate(f.readlines()):
        words = line.split(' ')
        words[-1] = words[-1].strip()
        # 训练doc2vec不应当去除停用词
        # print(' '.join(words))
        # words = filter(lambda x: x not in stopwords, words)
        # print(' '.join(words))
        x_train.append(TaggededDocument(words, tags=[i]))
        if i % 100000 == 0:
            print('processed %s/%s' % (i, 20000000))
            if i == 5000000:
                break
    return x_train


def train_dm(x_train):
    model_dm = Doc2Vec(x_train, min_count=5, window=5, size=200, sample=1e-3, negative=5, workers=8)
    model_dm.train(x_train, total_examples=model_dm.corpus_count, epochs=10)
    model_dm.save('./model/dm')


def train_dbow(x_train):
    model_dbow = Doc2Vec(x_train, min_count=5, window=5, size=200, sample=1e-3, negative=5, workers=8, dbow_words=1,
                         dm=0)
    model_dbow.train(x_train, total_examples=model_dbow.corpus_count, epochs=10)
    model_dbow.save('./model/dbow')


def train():
    x_train = get_dataset()
    train_dm(x_train)
    train_dbow(x_train)


def cos_dist(vec1, vec2):
    vector1 = np.array(vec1)
    vector2 = np.array(vec2)
    return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * (np.linalg.norm(vector2)))


model = Doc2Vec.load('./model/dm')


def get_sentence_vectors(sentence):
    return model.infer_vector(list(jieba.cut(sentence)))


sentences = [u'用大蒜鉴别地沟油的方法,怎么鉴别地沟油', u'一颗大蒜就能鉴别地沟油？别闹了！做到下面几点，让您远离地沟油',
             u'飞机就要起飞，一个男人在机舱口跪下！', u'#吃秀##美好的一天从早餐开始#续集这会灯好多，上个一会黑一会白本来皮肤就黄',
             u'#营养美食季#糖尿病人到底能不能吃土豆？', u'六月蚊子多多，千万别用蚊香！教你一招，家里屋子的蚊子全都死光',
             u'6月蚊子无法无天！蚊香无用不环保，教你一招妙计，蠢蚊子哭死了', u'30岁的赵丽颖发博：我的爱，告白冯绍峰？冯绍峰只回了8个字',
             u'刘谦回归登湖南卫视春晚，湖南卫视会让谁来当“董卿”?', u'2017春晚有沈腾吗', u'1万元十年后是现在的多少钱',
             u'53岁朱军即将退休，接替者小撒和小尼落选？你绝对想不到是他', u'53岁朱军即将退出央视 接替者撒贝宁和小尼落选？',
             u'吃防腐剂吧', u'桃打防腐剂', u'永动机', u'永动机~', u'白头发越来越多是为什么？', u'白头发越来越多了', u'号外！快来看这里有“龙”在空中飞',
             u'哇!真的有龙在空中飞']


def test():
    for s1 in sentences:
        for s2 in sentences:
            print(s1)
            print(s2)
            print(' '.join(list(jieba.cut(s1))))
            print(' '.join(list(jieba.cut(s2))))
            print(cos_dist(get_sentence_vectors(s1), get_sentence_vectors(s2)))
    for s in sentences:
        print(s)
        for k, v in model.most_similar([get_sentence_vectors(s)]):
            print(k)
            print(v)


def get_stopwords():
    stopwords = []
    for word in open('./data/stopwords.txt', 'r').readlines():
        word = word.strip()
        if word:
            stopwords.append(word)
    print('Total have %s stopwords' % len(stopwords))
    return stopwords


def test_jieba():
    for s in sentences:
        print(s + ' ' + (' '.join(jieba.analyse.extract_tags(s, topK=20, withWeight=False, allowPOS=allowPos))))
        print(s + ' ' + (' '.join(jieba.analyse.extract_tags(s, topK=20, withWeight=False))))
    df = pd.read_csv('./data/train.csv', encoding='utf8')
    tf = df[df.label == u'disagreed'][0:100]
    for ind, row in tf.iterrows():
        print(' '.join([row['title1'], row['title2'], row['label']]))
        s = row['title1']
        print(s + ' ' + (' '.join(jieba.analyse.extract_tags(s, topK=20, withWeight=False, allowPOS=allowPos))))
        print(s + ' ' + (' '.join(jieba.analyse.extract_tags(s, topK=20, withWeight=False))))
        s = row['title2']
        print(s + ' ' + (' '.join(jieba.analyse.extract_tags(s, topK=20, withWeight=False, allowPOS=allowPos))))
        print(s + ' ' + (' '.join(jieba.analyse.extract_tags(s, topK=20, withWeight=False))))


allowPos = ['n', 'nr', 'nr1', 'nr2', 'ns', 'nrj', 'nt', 'nsf', 'nz', 't', 'v', 'vn', 'm', 'nl', 'ng']


def jaccard_dist(x, y):
    xx = jieba.analyse.extract_tags(x, topK=20, withWeight=False, allowPOS=allowPos)
    yy = jieba.analyse.extract_tags(y, topK=20, withWeight=False, allowPOS=allowPos)
    # print(' '.join(xx))
    # print(' '.join(yy))
    try:
        return 1.0 * len(set(xx) & set(yy)) / len(set(xx) | set(yy))
    except Exception, e:
        print(e)
        print(x)
        print(y)
        print(xx)
        print(yy)
        print(' '.join(xx))
        print(' '.join(yy))
        return 0


def test_jaccard():
    for s1 in sentences:
        for s2 in sentences:
            print(s1 + ' ' + s2 + ' ' + str(jaccard_dist(s1, s2)))


if __name__ == '__main__':
    # train()
    # test()
    test_jieba()
    # test_jaccard()
    # print(jaccard_dist(u'六月蚊子多多，千万别用蚊香！教你一招，家里屋子的蚊子全都死光', u'6月蚊子无法无天！蚊香无用不环保，教你一招妙计，蠢蚊子哭死了'))
    # print(cos_dist([1, 1], [2, 2]), cos_dist([1, 0], [0, 1]))
