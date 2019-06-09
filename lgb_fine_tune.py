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

    lgb_train = lgb.Dataset(features, label)
    ### 设置初始参数--不含交叉验证参数
    print('设置参数')
    params = {
        'objective': 'multiclass',
        'nthread': 4,
        'learning_rate': 0.1,
        'num_class': 3
    }

    ### 交叉验证(调参)
    print('交叉验证')
    multi_logloss_min = float('0')
    best_params = {}
    # 准确率
    print("调参1：提高准确率")
    for num_leaves in range(5, 100, 5):
        for max_depth in range(3, 8, 1):
            params['num_leaves'] = num_leaves
            params['max_depth'] = max_depth

            cv_results = lgb.cv(
                params,
                lgb_train,
                seed=1,
                nfold=5,
                early_stopping_rounds=10,
                verbose_eval=True
            )
            multi_logloss_mean = pd.Series(cv_results['multi_logloss-mean']).min()
            boost_rounds = pd.Series(cv_results['multi_logloss-mean']).idxmin()

            if multi_logloss_mean <= multi_logloss_min:
                multi_logloss_min = multi_logloss_mean
                best_params['num_leaves'] = num_leaves
                best_params['max_depth'] = max_depth
    if 'num_leaves' and 'max_depth' in best_params.keys():
        params['num_leaves'] = best_params['num_leaves']
        params['max_depth'] = best_params['max_depth']

    # 过拟合
    print("调参2：降低过拟合")
    for max_bin in range(5, 256, 10):
        for min_data_in_leaf in range(1, 102, 10):
            params['max_bin'] = max_bin
            params['min_data_in_leaf'] = min_data_in_leaf

            cv_results = lgb.cv(
                params,
                lgb_train,
                seed=1,
                nfold=5,
                early_stopping_rounds=10,
                verbose_eval=True
            )

            multi_logloss_mean = pd.Series(cv_results['multi_logloss-mean']).min()
            boost_rounds = pd.Series(cv_results['multi_logloss-mean']).idxmin()

            if multi_logloss_mean <= multi_logloss_min:
                multi_logloss_min = multi_logloss_mean
                best_params['max_bin'] = max_bin
                best_params['min_data_in_leaf'] = min_data_in_leaf
    if 'max_bin' and 'min_data_in_leaf' in best_params.keys():
        params['min_data_in_leaf'] = best_params['min_data_in_leaf']
        params['max_bin'] = best_params['max_bin']

    print("调参3：降低过拟合")
    for feature_fraction in [0.6, 0.7, 0.8, 0.9, 1.0]:
        for bagging_fraction in [0.6, 0.7, 0.8, 0.9, 1.0]:
            for bagging_freq in range(0, 50, 5):
                params['feature_fraction'] = feature_fraction
                params['bagging_fraction'] = bagging_fraction
                params['bagging_freq'] = bagging_freq

                cv_results = lgb.cv(
                    params,
                    lgb_train,
                    seed=1,
                    nfold=5,
                    early_stopping_rounds=10,
                    verbose_eval=True
                )

                multi_logloss_mean = pd.Series(cv_results['multi_logloss-mean']).min()
                boost_rounds = pd.Series(cv_results['multi_logloss-mean']).idxmin()

                if multi_logloss_mean <= multi_logloss_min:
                    multi_logloss_min = multi_logloss_mean
                    best_params['feature_fraction'] = feature_fraction
                    best_params['bagging_fraction'] = bagging_fraction
                    best_params['bagging_freq'] = bagging_freq

    if 'feature_fraction' and 'bagging_fraction' and 'bagging_freq' in best_params.keys():
        params['feature_fraction'] = best_params['feature_fraction']
        params['bagging_fraction'] = best_params['bagging_fraction']
        params['bagging_freq'] = best_params['bagging_freq']

    print("调参4：降低过拟合")
    for lambda_l1 in [1e-5, 1e-3, 1e-1, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
        for lambda_l2 in [1e-5, 1e-3, 1e-1, 0.0, 0.1, 0.4, 0.6, 0.7, 0.9, 1.0]:
            params['lambda_l1'] = lambda_l1
            params['lambda_l2'] = lambda_l2
            cv_results = lgb.cv(
                params,
                lgb_train,
                seed=1,
                nfold=5,
                early_stopping_rounds=10,
                verbose_eval=True
            )

            multi_logloss_mean = pd.Series(cv_results['multi_logloss-mean']).min()
            boost_rounds = pd.Series(cv_results['multi_logloss-mean']).idxmin()

            if multi_logloss_mean <= multi_logloss_min:
                multi_logloss_min = multi_logloss_mean
                best_params['lambda_l1'] = lambda_l1
                best_params['lambda_l2'] = lambda_l2
    if 'lambda_l1' and 'lambda_l2' in best_params.keys():
        params['lambda_l1'] = best_params['lambda_l1']
        params['lambda_l2'] = best_params['lambda_l2']

    print("调参5：降低过拟合2")
    for min_split_gain in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        params['min_split_gain'] = min_split_gain

        cv_results = lgb.cv(
            params,
            lgb_train,
            seed=1,
            nfold=5,
            early_stopping_rounds=10,
            verbose_eval=True
        )

        multi_logloss_mean = pd.Series(cv_results['multi_logloss-mean']).min()
        boost_rounds = pd.Series(cv_results['multi_logloss-mean']).idxmin()

        if multi_logloss_mean <= multi_logloss_min:
            multi_logloss_min = multi_logloss_mean

            best_params['min_split_gain'] = min_split_gain
    if 'min_split_gain' in best_params.keys():
        params['min_split_gain'] = best_params['min_split_gain']
    print(params)

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


    """
    result:
    {'num_leaves': 95, 'num_class': 3, 'lambda_l1': 1.0, 'bagging_freq': 45, 'learning_rate': 0.1, 'lambda_l2': 1.0, 'nthread': 4, 'min_split_gain': 1.0, 'min_data_in_leaf': 101, 'max_bin': 255, 'objective': 'multiclass', 'bagging_fraction': 1.0, 'max_depth': 7, 'feature_fraction': 1.0}
    """
