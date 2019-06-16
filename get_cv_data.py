#encoding=utf8
import pandas as pd
from collections import defaultdict

if __name__ == '__main__':
    train = pd.read_csv('./data/train_clean.csv')
    train = train.sample(frac=1)
    train['hash'] = train.apply(lambda row: (hash(row['title1']) % 5 + 5) % 5, axis=1)
    columns = ['id', 'title1', 'title2', 'label']
    train[columns].to_csv('./data/train.tsv', index=False, header=False, sep='\t')
    for cv in range(5):
        this_train = train[train.hash != cv][columns]
        this_eval = train[train.hash == cv][columns]
        this_train.to_csv('./data/train_%s.csv' % cv, index=False, header=False)
        this_train.to_csv('./data/train_%s.tsv' % cv, index=False, header=False, sep='\t')
        this_eval.to_csv('./data/eval_%s.csv' % cv, index=False, header=False)
        this_eval.to_csv('./data/eval_%s.tsv' % cv, index=False, header=False, sep='\t')
        print('this train has %s agree, %s unrelated, %s disagree' % (len(this_train[this_train.label == 'agreed']),
                                                                      len(this_train[this_train.label == 'unrelated']),
                                                                      len(this_train[this_train.label == 'disagreed'])))
    test = pd.read_csv('data/test.csv')
    print('initially we have %s test data' % len(test))
    test = pd.concat([test, train[['id', 'title1', 'title2']]])
    test = test.sample(frac=1)
    print('add train_data we have %s' % len(test))
    test.to_csv('data/test.tsv', sep='\t', index=False, header=False)
