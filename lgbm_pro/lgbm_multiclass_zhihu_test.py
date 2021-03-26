#!usr/bin/env python
#-*- coding:utf-8 _*-
"""
@version: python3.6
@author: QLMX
@contact: wenruichn@gmail.com
@time: 2019-07-29 08:56
公众号：AI成长社
知乎：https://www.zhihu.com/people/qlmx-61/columns
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from sklearn.metrics import roc_auc_score,accuracy_score

category = 27
## load data
train_data = pd.read_csv('./temp_train.txt',header=None,sep=' ')
test_data = pd.read_csv('./temp_test.txt',header=None,sep=' ')
num_round = 1000

def pre_deal_data(x):
    if type(x) != int:
        feature_list = x.split(":")
    else:
        feature_list = [x]
    if len(feature_list) > 1:
        return float(feature_list[1])
    else:
        return float(x)

train_data = train_data.applymap(pre_deal_data)

test_data = test_data.applymap(pre_deal_data)

## category feature one_hot
# test_data['flag'] = -1
# train_data['flag'] = 1
# data = pd.concat([train_data, test_data])
# cate_feature = ['gender', 'cell_province', 'id_province', 'id_city', 'rate', 'term']
# for item in cate_feature:
#     data[item] = LabelEncoder().fit_transform(data[item])

# train = data[data['flag'] != -1]
# test = data[data['flag'] == -1]

##Clean up the memory
# del data, train_data, test_data
# gc.collect()

## get train feature
# del_feature = ['auditing_date', 'due_date', 'label','flag']
# features = [i for i in train.columns if i not in del_feature]

train_x = train_data.drop(0,axis=1,inplace=False)
train_y = train_data[0].astype(int)
test_x = test_data.drop(0,axis=1,inplace=False)
test_y = test_data[0].astype(int)


params = {'num_leaves': 60,
          'min_data_in_leaf': 30,
          'objective': 'multiclass',
          'num_class': category,
          'max_depth': -1,
          'learning_rate': 0.03,
          "min_sum_hessian_in_leaf": 6,
          "boosting": "gbdt",
          "feature_fraction": 0.9,
          "bagging_freq": 1,
          "bagging_fraction": 0.8,
          "bagging_seed": 11,
          "lambda_l1": 0.1,
          "verbosity": -1,
          "nthread": 15,
          'metric': 'multi_logloss',
          "random_state": 2019,
          # 'device': 'gpu'
          }


folds = KFold(n_splits=5, shuffle=True, random_state=2019)
prob_oof = np.zeros((train_x.shape[0], category))
test_pred_prob = np.zeros((test_x.shape[0], category))

## train and predict
feature_importance_df = pd.DataFrame()
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_data)):
    print("fold {}".format(fold_ + 1))
    trn_data = lgb.Dataset(train_x.iloc[trn_idx], label=train_y.iloc[trn_idx])
    val_data = lgb.Dataset(train_x.iloc[val_idx], label=train_y.iloc[val_idx])

    clf = lgb.train(params,
                    trn_data,
                    num_round,
                    valid_sets=[trn_data, val_data],
                    verbose_eval=20,
                    # categorical_feature=None,
                    early_stopping_rounds=60)
    prob_oof[val_idx] = clf.predict(train_x.iloc[val_idx], num_iteration=clf.best_iteration)


    # fold_importance_df = pd.DataFrame()
    # fold_importance_df["Feature"] = features
    # fold_importance_df["importance"] = clf.feature_importance()
    # fold_importance_df["fold"] = fold_ + 1
    # feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    test_pred_prob += clf.predict(test_x, num_iteration=clf.best_iteration) / folds.n_splits

result = np.argmax(test_pred_prob, axis=1)


for i in range(len(test_y)):
    print("%s:%s"%(test_y[i],result[i]))

print(accuracy_score(test_y,result))   #该部分中由于数据的测试集中没有label标签，所以无法做测试集的验证计算准确率

# predictions = []
# for x in test_pred_prob:
#     predictions.append(np.argmax(x))
#
# print(predictions)

## plot feature importance
# cols = (feature_importance_df[["Feature", "importance"]].groupby("Feature").mean().sort_values(by="importance", ascending=False).index)
# best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)].sort_values(by='importance',ascending=False)
# plt.figure(figsize=(8, 10))
# sns.barplot(y="Feature",
#             x="importance",
#             data=best_features.sort_values(by="importance", ascending=False))
# plt.title('LightGBM Features (avg over folds)')
# plt.tight_layout()
# plt.savefig('./dmm/result/lgb_importances.png')