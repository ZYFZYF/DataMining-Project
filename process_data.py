import pandas as pd
from sklearn.model_selection import train_test_split


def format_label(x):
    return {'disagreed': -1,
            'unrelated': 0,
            'agreed': 1}.get(x, x)


if __name__ == '__main__':
    df = pd.read_csv('data/train.csv')
    train, dev, train_label, dev_label = train_test_split(df[['id', 'title1', 'title2']],
                                                          df['label'],
                                                          test_size=0.2,
                                                          random_state=42)
    train['label'] = train_label
    dev['label'] = dev_label
    test = pd.read_csv('data/test.csv')
    train.to_csv('data/train.tsv', sep='\t', index=False)
    dev.to_csv('data/dev.tsv', sep='\t', index=False)
    test.to_csv('data/test.tsv', sep='\t', index=False)
