#encoding=utf8
from utils_common import CV_NUMS
import pandas as pd


def get_max_index(row):
    poss = [row['poss_a'], row['poss_b'], row['poss_c']]
    return ['agreed', 'unrelated', 'disagreed'][poss.index(max(poss))]


if __name__ == '__main__':
    for cv in range(CV_NUMS):
        df = pd.read_csv('./data/test_%s.tsv' % cv, sep='\t', names=['id', 'poss_a', 'poss_b', 'poss_c'])
        print(len(df))
        df = df[(df.id == 321187) | (df.id > 321188)]
        df.sort_values(by=['id'], inplace=True)
        print(len(df))
        df['result'] = df.apply(get_max_index, axis=1)
        df = df[['id', 'result']]
        # print(ans['result'])
        print(len(df))
        tf = pd.read_csv('./data/test.csv')
        old_ids = set(list(tf['id'].to_list()))
        now_ids = set(list(df['id'].to_list()))
        print(len(old_ids), len(now_ids), old_ids - now_ids)
        df.loc[len(df)] = ['378161', 'unrelated']
        df.loc[len(df)] = ['329037', 'disagreed']
        print(len(df))
        df.to_csv('./data/result_%s.txt' % cv, index=False, header=False, sep='\t')

