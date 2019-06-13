#encoding=utf8
import pandas as pd


def get_max_index(row):
    poss = [row['poss_a'], row['poss_b'], row['poss_c']]
    return ['agreed', 'unrelated', 'disagreed'][poss.index(max(poss))]


if __name__ == '__main__':
    df = pd.read_csv('./data/test_results_dup.tsv', sep='\t', names=['id', 'poss_a', 'poss_b', 'poss_c'])
    tf = pd.read_csv('./data/test.csv')
    tf.dropna(how='any', axis=0, inplace=True)
    print(len(df), len(tf))
    ans = pd.DataFrame(columns=['id', 'result'])
    ans['id'] = tf['id']
    ans['result'] = df.apply(get_max_index, axis=1)
    print(len(ans))
    # print(ans['result'])
    ans.loc[len(ans) + 1] = ['357062', 'unrelated']
    print(len(ans))
    ans[['id', 'result']].to_csv('./data/result.txt', index=False, header=False, sep='\t')