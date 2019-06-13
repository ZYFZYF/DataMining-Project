import pandas as pd
from sklearn.model_selection import train_test_split


def format_label(x):
    return {'disagreed': -1,
            'unrelated': 0,
            'agreed': 1}.get(x, x)


if __name__ == '__main__':
    train = pd.read_csv('data/train.csv')
    print('origin we have %s train data' % len(train))
    train.dropna(how='any', axis=0, inplace=True)
    print('after drop nan data, we have %s' % len(train))
    test = pd.read_csv('data/test.csv')
    print('initially we have %s test data' % len(test))
    test = pd.concat([test, train[['id', 'title1', 'title2']]])
    print('add train_data we have %s' % len(test))
    test.to_csv('data/test.tsv', sep='\t', index=False)
    test = pd.read_csv('data/test.csv')
    disagree_train_data = train[train.label == u'disagreed']
    print('we have %s disagree label data' % len(disagree_train_data))
    # train = pd.concat([train, disagree_train_data, disagree_train_data])
    print('after dup three times disagree data, we have %s' % len(train))
    train = train.sample(frac=1)
    train.to_csv('data/train.tsv', sep='\t', index=False)
