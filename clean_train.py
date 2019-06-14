#encoding=utf8
import pandas as pd
from collections import defaultdict

def my_hash(row):
    return (hash(row['title1']) % 5 + 5) % 5


def is_hanzi(uchar):
    return uchar >= u'\u4e00' and uchar <= u'\u9fa5'


labels = defaultdict(set)


def get_clean_text(row):
    text = unicode(row['title1'] + row['title2'], 'utf8')
    clean_text = filter(is_hanzi, text)
    labels[clean_text].add(row['label'])
    return clean_text


if __name__ == '__main__':
    df = pd.read_csv('./data/train.csv')
    print('initial we have %s' % len(df))
    df.dropna(how='any', axis=0, inplace=True)
    print('after drop na we have %s' % len(df))
    df['hash'] = df.apply(my_hash, axis=1)
    df['clean_text'] = df.apply(get_clean_text, axis=1)
    df['label_size'] = df.apply(lambda row: len(labels[row['clean_text']]), axis=1)
    df = df[df.label_size == 1]
    print('after drop contradiction we have %s' % len(df))
    df = df[['id', 'title1', 'title2', 'label']]
    df.to_csv('./data/train_clean.csv', index=False, encoding='utf8')
