TRAIN_FEATURES = ['keywords_jaccard_distance',
                  'title1_negative_count',
                  'title2_negative_count',
                  'keywords_meaning_distance_after_remove_negative_word']
LABEL = 'label'
TRAIN_FEATURES_NEW = [#'keywords_jaccard_distance',
                      #'title1_negative_count',
                      #'title2_negative_count',
                      #'keywords_meaning_distance_after_remove_negative_word',
                      'bert_prob_agree',
                      'bert_prob_unrelate',
                      'bert_prob_disagree'
                      # 'keywords_meaning_distance',
                      # 'keywords_jaccard_distance_after_remove_negative_word'
                      ]
CATEGORICAL_FEATURES = []#['title1_negative_count', 'title2_negative_count']
