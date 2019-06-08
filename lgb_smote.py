# coding:utf8
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn import metrics
from sklearn import model_selection
from utils_logging import Logging
from utils_common import TRAIN_FEATURES, LABEL, TRAIN_FEATURES_NEW
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, ADASYN
from collections import Counter, defaultdict
logging = Logging()


def format_label(x):
    return {u'agreed': 0,
            u'unrelated': 1,
            u'disagreed': 2}.get(x)


def label_format(x):
    return {0: u'agreed',
            1: u'unrelated',
            2: u'disagreed'}.get(x)


def online_score(ans, pre):
    ret = 0
    tot = 0
    data = defaultdict(int)
    right = defaultdict(int)
    for x, y in zip(ans, pre):
        if x == 1:
            rate = 1.0 / 16
        elif x == 0:
            rate = 1.0 / 15
        else:
            rate = 1.0 / 5
        if x == y:
            ret += rate
            right[x] += 1
        data[x] += 1
        tot += rate
    print(data)
    print(right)
    return ret / tot


if __name__ == '__main__':
    # 获取数据集
    data = pd.read_csv('data/train_after_clean_after_process.csv')
    train_data = data[data.label == data.label]
    agree_data = train_data[train_data.label == u'agreed']
    disagree_data = train_data[train_data.label == u'disagreed']
    unrelated_data = train_data[train_data.label == u'unrelated']
    print(len(disagree_data), len(agree_data), len(unrelated_data))
    print(len(train_data))
    logging.info("we have %s train datas" % len(train_data))
    train_data.dropna(how='any', axis=0, inplace=True)
    features = train_data[TRAIN_FEATURES_NEW]
    new_features = features.dropna(how='any', axis=0, inplace=False)
    label = train_data[LABEL].apply(format_label)
    X_train, X_eval, Y_train, Y_eval = train_test_split(features, label, test_size=0.2, random_state=42)
    print(sorted(Counter(Y_train).items()))
    X_train, Y_train = SMOTE().fit_sample(X_train, Y_train)
    print(sorted(Counter(Y_train).items()))
    train_data = lgb.Dataset(X_train, label=Y_train)
    validation_data = lgb.Dataset(X_eval, label=Y_eval)
    params = {
        'learning_rate': 0.1,
        'lambda_l1': 0.1,
        'lambda_l2': 0.2,
        'max_depth': 5,
        'objective': 'multiclass',
        'num_class': 3,
        'num_leaves': 20,
        'min_data_in_leaf': 41,
        'max_bin': 95,
        'bagging_freq': 0,
        'bagging_fraction': 0.6,
        'feature_fraction': 0.8,
        # 'num_boost_round': 10000,# 没有区别
        # 'early_stopping_rounds': 50
        # lightgbm.basic.LightGBMError: b‘Number of classes should be specified and greater than 1 for multiclass training‘
    }
    clf = lgb.train(params, train_data, valid_sets=[validation_data])
    from sklearn.metrics import roc_auc_score, accuracy_score

    y_pred = clf.predict(X_eval)
    y_pred = [list(x).index(max(x)) for x in y_pred]

    # print(y_pred)
    logging.info('auc is %s' % accuracy_score(Y_eval, y_pred))
    logging.info('online score is %s' % online_score(Y_eval, y_pred))
    test_data = data[data.label != data.label]
    X_test = test_data[TRAIN_FEATURES_NEW]
    logging.info("we need to predict %s test datas" % len(X_test))
    y_pred = clf.predict(X_test)
    y_pred = [list(x).index(max(x)) for x in y_pred]
    sub = test_data.copy()
    sub['label'] = y_pred
    sub['label'] = sub['label'].apply(label_format)
    sub = sub[['id', 'label']]
    sub.loc[len(sub)] = [357062, u'unrelated']
    sub[['id', 'label']].to_csv('./data/result.txt', index=False, header=False, sep='\t')
    print(clf.feature_importance())
    print(len(sub[sub.label == u'agreed']),
          len(sub[sub.label == u'disagreed']),
          len(sub[sub.label == u'unrelated']))
