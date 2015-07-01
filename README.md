# Ali2015-MobileRecommendation
Ali Mobile Recommendation Algorithm Competition - http://tianchi.aliyun.com/competition/introduction.htm

1. Run `TianChi3/main_preprocess.py` once to pre-process the data

2. Run `TianChi3/main_single_model.py` step by step to tune and train the models.

----

For the best single GBDT model, the F1 score is around 10.4%, which has a rank about 100 in Season 1.

To improve the performance
    1. By changing the time parameters in utils.gen_feats, and utils.gen_ic_ind_feats, more labeled data can be generated, which is useful because the dataset is highly unbalanced.
    2. Add more time intervals in utils.gen_feats, and utils.gen_ic_ind_feats
    3. Cross-validation can be used to select better hyper-parameters.
    4. Model ensemble can be used. A possilbe method:
        1) Train multiple 'not bad' single models with different algorithms (logistic regression, GBDT, adaboost, randomforest...)
        2) Use single models' outputs as input to train a new logistic regression
        3) Use the outer logistic regression to make the predic