TRAIN_FEATURES = ['keywords_jaccard_distance',
                  'title1_negative_count',
                  'title2_negative_count',
                  'keywords_meaning_distance_after_remove_negative_word']
LABEL = 'label'
# TRAIN_FEATURES_NEW = [  # 'keywords_jaccard_distance',
#     # 'title1_negative_count',
#     # 'title2_negative_count',
#     # 'keywords_meaning_distance_after_remove_negative_word',
#     'bert_0_prob_agree',
#     'bert_0_prob_unrelate',
#     'bert_0_prob_disagree',
#     'bert_1_prob_agree',
#     'bert_1_prob_unrelate',
#     'bert_1_prob_disagree',
#     'bert_2_prob_agree',
#     'bert_2_prob_unrelate',
#     'bert_2_prob_disagree',
#     'bert_3_prob_agree',
#     'bert_3_prob_unrelate',
#     'bert_3_prob_disagree',
#     'bert_4_prob_agree',
#     'bert_4_prob_unrelate',
#     'bert_4_prob_disagree',
#     'bert_5_prob_agree',
#     'bert_5_prob_unrelate',
#     'bert_5_prob_disagree',
#     'bert_6_prob_agree',
#     'bert_6_prob_unrelate',
#     'bert_6_prob_disagree',
#     'bert_7_prob_agree',
#     'bert_7_prob_unrelate',
#     'bert_7_prob_disagree',
#     'bert_8_prob_agree',
#     'bert_8_prob_unrelate',
#     'bert_8_prob_disagree',
#     'bert_9_prob_agree',
#     'bert_9_prob_unrelate',
#     'bert_9_prob_disagree',
#     # 'keywords_meaning_distance',
#     # 'keywords_jaccard_distance_after_remove_negative_word'
# ]
CATEGORICAL_FEATURES = []  # ['title1_negative_count', 'title2_negative_count']
CV_NUMS = 10
TRAIN_FEATURES_NEW = ['bert_%s_prob_disagree' % i for i in range(CV_NUMS)] + ['bert_%s_prob_agree' % i for i in
                                                                              range(CV_NUMS)] + [
                         'bert_%s_prob_unrelate' % i for i in range(CV_NUMS)]
