# coding:utf8
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn import metrics
from sklearn import model_selection
from utils_logging import Logging
from utils_common import TRAIN_FEATURES, LABEL, CATEGORICAL_FEATURES
from sklearn.model_selection import train_test_split
from collections import defaultdict

logging = Logging()


def format_label(x):
    return {u'agreed': 0,
            u'unrelated': 1,
            u'disagreed': 2}.get(x)


def label_format(x):
    return {0: u'agreed',
            1: u'unrelated',
            2: u'disagreed'}.get(x)


def online_score(ans, pre, eval_set):
    wrong = eval_set.copy()
    wrong['true_label'] = map(label_format, ans)
    wrong['pred_label'] = map(label_format, pre)
    wrong = wrong[wrong.true_label != wrong.pred_label]
    wrong.sort_values(by=['true_label', 'pred_label'], inplace=True)
    wrong.to_csv('./data/wrong_case.csv', encoding='utf8', index=False)
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
    data = pd.read_csv('data/train_categorical_final.csv')
    for col in CATEGORICAL_FEATURES:
        data[col] = data[col].astype('category')
    TRAIN_FEATURES_NEW = ['bert_prob_agree', 'bert_prob_unrelate', 'bert_prob_disagree']
    bert_results = pd.read_csv('data/test_all.tsv', sep='\t',
                               names=['id', 'bert_prob_agree', 'bert_prob_unrelate', 'bert_prob_disagree'])
    data = pd.merge(left=data, right=bert_results, on='id', how='left')
    print(data.head)
    print(data.columns)
    train_data = data[data.label == data.label]
    print(len(train_data[train_data.label == u'disagreed']), len(train_data[train_data.label == u'agreed']),
          len(train_data[train_data.label == u'unrelated']))
    print(len(train_data))
    print(train_data.head)
    train_data.dropna(how='any', axis=0, inplace=True)

    logging.info("we have %s train datas" % len(train_data))
    features = train_data
    label = train_data[LABEL].apply(format_label)
    X_train, X_eval, Y_train, Y_eval = train_test_split(features, label, test_size=0.2, random_state=42, shuffle=True)
    X_train = X_train[TRAIN_FEATURES_NEW]
    temp = X_train.copy()
    temp[LABEL] = Y_train
    agree_data = temp[temp.label == 0]
    disagree_data = temp[temp.label == 2]
    unrelated_data = temp[temp.label == 1]
    temp = pd.concat([temp, disagree_data, disagree_data])
    # temp = pd.concat([temp])
    X_train = temp[TRAIN_FEATURES_NEW]
    Y_train = temp[LABEL]
    print(len(X_train), len(agree_data), len(disagree_data), len(unrelated_data))
    print(X_train.columns)
    train_data = lgb.Dataset(X_train, label=Y_train)
    validation_data = lgb.Dataset(X_eval[TRAIN_FEATURES_NEW], label=Y_eval)
    params = {
        'objective': 'multiclass',
        'lambda_l1': 0.1,
        'lambda_l2': 0.2,
        'learning_rate': 0.01,
        'num_class': 3,
        'num_boost_round': 50000,  # 没有区别
        'early_stopping_rounds': 50,
        # 'num_leaves': 95,
        # 'bagging_freq': 45,
        # 'nthread': 4,
        # 'min_split_gain': 1.0,
        # 'min_data_in_leaf': 101,
        # 'max_bin': 255,
        # 'bagging_fraction': 1.0,
        # 'max_depth': 7,
        # 'feature_fraction': 1.0
        # lightgbm.basic.LightGBMError: b‘Number of classes should be specified and greater than 1 for multiclass training‘
    }
    clf = lgb.train(params, train_data, valid_sets=[validation_data])
    from sklearn.metrics import roc_auc_score, accuracy_score

    y_pred = clf.predict(X_eval[TRAIN_FEATURES_NEW])
    y_pred = [list(x).index(max(x)) for x in y_pred]

    # print(y_pred)
    logging.info('auc is %s' % accuracy_score(Y_eval, y_pred))
    logging.info('online score is %s' % online_score(Y_eval, y_pred, X_eval))
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
    # sub.loc[len(sub)] = ['378161', 'unrelated']
    # sub.loc[len(sub)] = ['329037', 'disagreed']
    sub[['id', 'label']].to_csv('./data/result.txt', index=False, header=False, sep='\t')
    print(clf.feature_importance())
    print(len(sub[sub.label == u'agreed']),
          len(sub[sub.label == u'disagreed']),
          len(sub[sub.label == u'unrelated']))
