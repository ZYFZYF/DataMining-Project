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
import jieba.posseg

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

TaggededDocument = gensim.models.doc2vec.TaggedDocument


def get_dataset():
    f = open('./data/reduce_titles.txt', 'r')
    x_train = []
    # stopwords = get_stopwords()
    for i, line in enumerate(f.readlines()):
        words = line.split(' ')
        words[-1] = words[-1].strip()
        print(words)
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
    return model.infer_vector([word.encode('utf8') if isinstance(word, unicode) else word for word in list(jieba.cut(sentence))])


sentences = [u'用大蒜鉴别地沟油的方法,怎么鉴别地沟油', u'一颗大蒜就能鉴别地沟油？别闹了！做到下面几点，让您远离地沟油',
             u'飞机就要起飞，一个男人在机舱口跪下！', u'#吃秀##美好的一天从早餐开始#续集这会灯好多，上个一会黑一会白本来皮肤就黄',
             u'#营养美食季#糖尿病人到底能不能吃土豆？', u'六月蚊子多多，千万别用蚊香！教你一招，家里屋子的蚊子全都死光',
             u'6月蚊子无法无天！蚊香无用不环保，教你一招妙计，蠢蚊子哭死了', u'30岁的赵丽颖发博：我的爱，告白冯绍峰？冯绍峰只回了8个字',
             u'刘谦回归登湖南卫视春晚，湖南卫视会让谁来当“董卿”?', u'2017春晚有沈腾吗', u'1万元十年后是现在的多少钱',
             u'53岁朱军即将退休，接替者小撒和小尼落选？你绝对想不到是他', u'53岁朱军即将退出央视 接替者撒贝宁和小尼落选？',
             u'吃防腐剂吧', u'桃打防腐剂', u'永动机', u'永动机~', u'白头发越来越多是为什么？', u'白头发越来越多了', u'号外！快来看这里有“龙”在空中飞',
             u'哇!真的有龙在空中飞', u'快看!天上有UFO在飞！', u'年轻必须吃点苦', u'年轻必须能吃苦', u'年轻必须能吃"苦"',
             u'猫吃鱼最健康？低级谣言最误人', u'“酒”混着喝易醉，“茶”混着喝会怎样？', u'衣冠禽兽原来是褒义词?', u'辟谣｜WIFI真的能杀精？',
             u'IQ越高，睡得越晚', u'5月谣言榜来袭', u'造谣"大连交警撵死人"被拘', u'盐吃多了会变丑？变傻？居然是真的！',
             u'褚时健老人为什们会被多次误传去世？', u'甄子丹搂景甜遭拒，真应学鹿晗、陈伟霆的“绅士手”', u'甄子丹对景甜做了一个小动作，却遭拒绝，场面十分尴尬',
             u'岳阳三荷机场试飞成功', u'岳阳市三荷机场客运飞机试飞成功', u'在藏区,看见白色帐篷别进去,有可能就成藏民的女婿了!', u'去西藏,路边的神秘的白帐篷可别随便乱进哦,小心失身!',
             u'紫薯是转基因吗？是染色吗？谣言止于此，告诉您不一样的紫薯！']


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

        s = s.replace('"', ' ').replace(u'“', ' ').replace(u'”', ' ')
        print(s + ' ' + (' '.join(jieba.analyse.extract_tags(s, topK=20, withWeight=False, allowPOS=allowPos))))
        print(s + ' ' + (' '.join(jieba.analyse.extract_tags(s, topK=20, withWeight=False))))
        words = jieba.posseg.cut(s)
        for w in words:
            print w.word, w.flag
    # df = pd.read_csv('./data/train.csv', encoding='utf8')
    # tf = df[df.label == u'disagreed'][0:100]
    # for ind, row in tf.iterrows():
    #     print(' '.join([row['title1'], row['title2'], row['label']]))
    #     s = row['title1']
    #     print(s + ' ' + (' '.join(jieba.analyse.extract_tags(s, topK=20, withWeight=False, allowPOS=allowPos))))
    #     print(s + ' ' + (' '.join(jieba.analyse.extract_tags(s, topK=20, withWeight=False))))
    #     s = row['title2']
    #     print(s + ' ' + (' '.join(jieba.analyse.extract_tags(s, topK=20, withWeight=False, allowPOS=allowPos))))
    #     print(s + ' ' + (' '.join(jieba.analyse.extract_tags(s, topK=20, withWeight=False))))


allowPos = ['n', 'nr', 'nr1', 'nr2', 'ns', 'nt', 'nz', 'nl', 'ng', 's', 't', 'v', 'vd', 'vb', 'eng', 'a', 'i', 'd', 'j']


def jaccard_dist_between_list(xx, yy):
    # return 1.0 * len(set(xx) & set(yy)) / len(set(xx) | set(yy))
    return 1.0 * len(set(xx) & set(yy)) / min(len(set(xx)), len(set(yy)))


def jaccard_dist(x, y):
    xx = jieba.analyse.extract_tags(x, topK=20, withWeight=False, allowPOS=allowPos)
    yy = jieba.analyse.extract_tags(y, topK=20, withWeight=False, allowPOS=allowPos)
    # print(' '.join(xx))
    # print(' '.join(yy))
    return jaccard_dist_between_list(xx, yy)


def word2vec_dist(x, y):
    return cos_dist(get_sentence_vectors(x), get_sentence_vectors(y))


def test_jaccard():
    for s1 in sentences:
        for s2 in sentences:
            print(s1 + ' ' + s2 + ' ' + str(jaccard_dist(s1, s2)))


def test_doc2vec(test_text):
    x_train = get_dataset()

    model_dm = Doc2Vec.load("./model/dm")
    inferred_vector_dm = model_dm.infer_vector(test_text)
    print inferred_vector_dm
    sims = model_dm.docvecs.most_similar([inferred_vector_dm], topn=10)
    for count, sim in sims:
        sentence = x_train[count]
        words = ''
        for word in sentence[0]:
            words = words + word + ' '
        print words, sim, len(sentence[0])


if __name__ == '__main__':
    train()
    # test()
    # test_jieba()
    # test_doc2vec(['《', '舞林', '争霸' '》', '十强' '出炉', '复活', '舞者', '澳门', '踢馆'])
    # test_doc2vec(['我国', '又一', '军舰', '顺利', '完成', '海试', '或', '将', '年底', '入伍', '专家', '意义', '堪比', '航母'])
    # import pdb;pdb.set_trace()
    # test_jaccard()
    # print(jaccard_dist(u'六月蚊子多多，千万别用蚊香！教你一招，家里屋子的蚊子全都死光', u'6月蚊子无法无天！蚊香无用不环保，教你一招妙计，蠢蚊子哭死了'))
    # print(cos_dist([1, 1], [2, 2]), cos_dist([1, 0], [0, 1]))
