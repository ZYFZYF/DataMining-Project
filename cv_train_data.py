#encoding=utf8
import pandas as pd


def my_hash(row):
    return (hash(row['title1']) % 5 + 5) % 5


if __name__ == '__main__':
    df = pd.read_csv('./data/train.csv')
    df.dropna(how='any', axis=0, inplace=True)
    df['hash'] = df.apply(my_hash, axis=1)
    print(df)