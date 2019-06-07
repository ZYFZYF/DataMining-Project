# coding:utf8
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn import metrics
from sklearn import model_selection
from utils_logging import Logging
from utils_common import TRAIN_FEATURES, LABEL, TRAIN_DATA_COUNT
from sklearn.model_selection import train_test_split

logging = Logging()


def format_label(x):
    return {u'agreed': 0,
            u'unrelated': 1,
            u'disagreed': 2}.get(x)


def label_format(x):
    return {0: u'agreed',
            1: u'unrelated',
            2: u'disagreed'}.get(x)


if __name__ == '__main__':
    # 获取数据集
    data = pd.read_csv('data/train_final.csv')
    train_data = data[data.label == data.label]
    logging.info("we have %s train datas" % len(train_data))
    features = train_data[TRAIN_FEATURES]
    label = train_data[LABEL].apply(format_label)
    X_train, X_eval, Y_train, Y_eval = train_test_split(features, label, test_size=0.2, random_state=42)
    train_data = lgb.Dataset(X_train, label=Y_train)
    validation_data = lgb.Dataset(X_eval, label=Y_eval)
    params = {
        'learning_rate': 0.1,
        'lambda_l1': 0.1,
        'lambda_l2': 0.2,
        'max_depth': 4,
        'objective': 'multiclass',
        'num_class': 3,
        # lightgbm.basic.LightGBMError: b‘Number of classes should be specified and greater than 1 for multiclass training‘
    }
    clf = lgb.train(params, train_data, valid_sets=[validation_data])
    from sklearn.metrics import roc_auc_score, accuracy_score

    y_pred = clf.predict(X_eval)
    y_pred = [list(x).index(max(x)) for x in y_pred]

    print(y_pred)
    print(accuracy_score(Y_eval, y_pred))

    test_data = data[data.label != data.label]
    X_test = test_data[TRAIN_FEATURES]
    logging.info("we need to predict %s test datas" % len(X_test))
    y_pred = clf.predict(X_test)
    y_pred = [list(x).index(max(x)) for x in y_pred]
    sub = test_data.copy()
    sub['label'] = y_pred
    sub['label'] = sub['label'].apply(label_format)
    sub = sub[['id', 'label']]
    sub.loc[len(sub)] = [357062, u'unrelated']
    sub[['id', 'label']].to_csv('./data/result.txt', index=False, header=False, sep='\t')
