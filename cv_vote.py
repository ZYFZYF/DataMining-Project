# coding=utf8
import pandas as pd
from collections import defaultdict

votes = defaultdict(dict)


def do_vote(row):
    if row['label'] in votes[row['id']].keys():
        votes[row['id']][row['label']] += 1
    else:
        votes[row['id']][row['label']] = 1
    return 0


if __name__ == '__main__':
    for i in range(5):
        df = pd.read_csv('./data/result_%s.txt' % i, names=['id', 'label'], sep='\t')
        df = df.apply(do_vote, axis=1)
    id_list = []
    label_list = []
    cnt = 0
    for k, v in votes.items():
        max_vote = 0
        max_label = 0
        for kk, vv in v.items():
            if vv > max_vote:
                max_vote = vv
                max_label = kk
        if len(v) != 1:
            cnt += 1
            print(v, max_vote, max_label)
        id_list.append(k)
        label_list.append(max_label)
    print('we have %s different results' % cnt)
    df = pd.DataFrame(data={'id': id_list, 'label': label_list})
    df.sort_values(by=['id'], inplace=True)
    df.to_csv('./data/cv_vote_result.txt', index=False, header=False, sep='\t')
