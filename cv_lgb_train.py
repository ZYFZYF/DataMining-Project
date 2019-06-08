# coding:utf8
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn import metrics
from sklearn import model_selection
from utils_logging import Logging
from utils_common import TRAIN_FEATURES, LABEL
from sklearn.model_selection import train_test_split, GridSearchCV

logging = Logging()


def format_label(x):
    return {u'agreed': 0,
            u'unrelated': 1,
            u'disagreed': 2}.get(x)


def label_format(x):
    return {0: u'agreed',
            1: u'unrelated',
            2: u'disagreed'}.get(x)


def give_acc(params):
    clf = lgb.train(params, train_data, valid_sets=[validation_data])
    y_pred = clf.predict(X_eval)
    y_pred = [list(x).index(max(x)) for x in y_pred]
    from sklearn.metrics import accuracy_score
    print(y_pred)
    print(accuracy_score(Y_eval, y_pred))


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
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'nthread': 4,
        'learning_rate': 0.1,
        'num_leaves': 30,
        'max_depth': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'num_class': 3
    }

    logging.start('train model')
    # fine-tune n_estimators result: {'n_estimators': 180
    """
    data_train = lgb.Dataset(X_train, Y_train)
    cv_results = lgb.cv(params, data_train, num_boost_round=1000, nfold=5, stratified=False, shuffle=True,
                        early_stopping_rounds=50, seed=0)
    logging.end()
    print(cv_results)
    print('best n_estimators:', len(cv_results['multi_logloss-mean']))
    print('best cv score:', pd.Series(cv_results['multi_logloss-mean']).max())
    params['n_estimators'] = len(cv_results['multi_logloss-mean'])
    logging.info("we need to predict %s test datas" % len(X_test))
    give_acc(params)
    """
    # fine-tune max_depth and num_leaves  result: {'num_leaves': 20, 'max_depth': 5} 这不是默认值么草
    """
    logging.start('fine tune')
    params_test1 = {'max_depth': range(3, 8, 1), 'num_leaves': range(5, 100, 5)}

    gsearch1 = GridSearchCV(
        estimator=lgb.LGBMClassifier(objective='multiclass', learning_rate=0.1,
                                     n_estimators=180, bagging_fraction=0.8, feature_fraction=0.8),
        param_grid=params_test1, scoring='accuracy', cv=5, n_jobs=-1)
    gsearch1.fit(X_train, Y_train)
    logging.end()
    print(gsearch1.cv_results_)
    best_score = 0
    for param, score in zip(gsearch1.cv_results_.get('params'), gsearch1.cv_results_.get('mean_test_score')):
        print(param, score)
        if score > best_score:
            best_score = score
            best_param = param
    import pdb;pdb.set_trace()
    """
    # fine-tune max_bin and min_data_in_leaf result: {'min_data_in_leaf':41, 'max_bin': 95}
    """
    logging.start('fine tune')
    params_test1 = {'max_bin': range(5, 256, 10), 'min_data_in_leaf': range(1, 102, 10)}

    gsearch1 = GridSearchCV(
        estimator=lgb.LGBMClassifier(objective='multiclass', learning_rate=0.1, n_estimators=180, num_leaves=20,
                                     max_depth=5, bagging_fraction=0.8, feature_fraction=0.8),
        param_grid=params_test1, scoring='accuracy', cv=5, n_jobs=-1)
    gsearch1.fit(X_train, Y_train)
    logging.end()
    best_score = 0
    for param, score in zip(gsearch1.cv_results_.get('params'), gsearch1.cv_results_.get('mean_test_score')):
        print(param, score)
        if score > best_score:
            best_score = score
            best_param = param
    import pdb;pdb.set_trace()
    """
    # fine-tune feature_fraction、bagging_fraction、bagging_freq  result: {'bagging_freq': 0, 'bagging_fraction': 0.6, 'feature_fraction': 0.8}
    """
    logging.start('fine tune')
    params_test1 = {'feature_fraction': [0.6, 0.7, 0.8, 0.9, 1.0],
                    'bagging_fraction': [0.6, 0.7, 0.8, 0.9, 1.0],
                    'bagging_freq': range(0, 81, 10)}

    gsearch1 = GridSearchCV(
        estimator=lgb.LGBMClassifier(objective='multiclass', learning_rate=0.1, n_estimators=180, num_leaves=20,
                                     max_depth=5, min_data_in_leaf=41, max_bin=95),
        param_grid=params_test1, scoring='accuracy', cv=5, n_jobs=-1)
    gsearch1.fit(X_train, Y_train)
    logging.end()
    best_score = 0
    for param, score in zip(gsearch1.cv_results_.get('params'), gsearch1.cv_results_.get('mean_test_score')):
        print(param, score)
        if score > best_score:
            best_score = score
            best_param = param
    import pdb;pdb.set_trace()
    """
    # fine-tune lambda_l1, lambda_l2 result: {'lambda_l1': 0.0, 'lambda_l2': 0.0} 这是个啥玩意儿啊
    """
    logging.start('fine tune')
    params_test1 = {'lambda_l1': [1e-5, 1e-3, 1e-1, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                    'lambda_l2': [1e-5, 1e-3, 1e-1, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]}

    gsearch1 = GridSearchCV(
        estimator=lgb.LGBMClassifier(objective='multiclass', learning_rate=0.1, n_estimators=180, num_leaves=20,
                                     max_depth=5, min_data_in_leaf=41, max_bin=95, bagging_freq=0, bagging_fraction=0.6,
                                     feature_fraction=0.8),
        param_grid=params_test1, scoring='accuracy', cv=5, n_jobs=-1)
    gsearch1.fit(X_train, Y_train)
    logging.end()
    best_score = 0
    for param, score in zip(gsearch1.cv_results_.get('params'), gsearch1.cv_results_.get('mean_test_score')):
        print(param, score)
        if score > best_score:
            best_score = score
            best_param = param
    import pdb;pdb.set_trace()
    """
    # fine-tune min_split_gain result:{'min_split_gain': 0.0} wtf?
    """
    logging.start('fine tune')
    params_test1 = {'min_split_gain': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}

    gsearch1 = GridSearchCV(
        estimator=lgb.LGBMClassifier(objective='multiclass', learning_rate=0.1, n_estimators=180, num_leaves=20,
                                     max_depth=5, min_data_in_leaf=41, max_bin=95, bagging_freq=0, bagging_fraction=0.6,
                                     feature_fraction=0.8, lambda_l1=0.0, lambda_l2=0.0),
        param_grid=params_test1, scoring='accuracy', cv=5, n_jobs=-1)
    gsearch1.fit(X_train, Y_train)
    logging.end()
    best_score = 0
    for param, score in zip(gsearch1.cv_results_.get('params'), gsearch1.cv_results_.get('mean_test_score')):
        print(param, score)
        if score > best_score:
            best_score = score
            best_param = param
    import pdb;

    pdb.set_trace()
    """

    clf = lgb.train(params, train_data, valid_sets=[validation_data])
    test_data = data[data.label != data.label]
    X_test = test_data[TRAIN_FEATURES]
    y_pred = clf.predict(X_test)
    y_pred = [list(x).index(max(x)) for x in y_pred]
    sub = test_data.copy()
    sub['label'] = y_pred
    sub['label'] = sub['label'].apply(label_format)
    sub = sub[['id', 'label']]
    sub.loc[len(sub)] = [357062, u'unrelated']
    sub[['id', 'label']].to_csv('./data/result.txt', index=False, header=False, sep='\t')
