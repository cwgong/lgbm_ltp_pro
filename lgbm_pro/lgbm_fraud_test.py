import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

import os

from sklearn import preprocessing
import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold

from matplotlib import pyplot as plt

train = pd.read_csv('./titanic_data/train.csv')
test = pd.read_csv('./titanic_data/test.csv')

# train_identity = pd.read_csv('../input/train_identity.csv', index_col='TransactionID')
# test_identity = pd.read_csv('../input/test_identity.csv', index_col='TransactionID')

# sample_submission = pd.read_csv('../input/sample_submission.csv', index_col='TransactionID')

print(train.head())

# train_identity.head()

# train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)
# test = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)

# print("shape of train is ....."+str(train.shape))
# print("shape of test is ....."+str(test.shape))

y_train = train['Survived'].copy()

# Drop target, fill in NaNs
X_train = train.drop('Survived', axis=1)
X_test = test.copy()
# X_train = X_train.fillna(-999)
# X_test = X_test.fillna(-999)
#
#
y_train = train['Survived'].copy()
plt.hist(y_train)
plt.title('isFraud distribution')
#
#reduce memory.
del train, test
#
# Label Encoding
for f in X_train.columns:
    if X_train[f].dtype=='object' or X_test[f].dtype=='object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(X_train[f].values) + list(X_test[f].values))
        X_train[f] = lbl.transform(list(X_train[f].values))
        X_test[f] = lbl.transform(list(X_test[f].values))

param = {'num_leaves': 120,
         'metric': 'auc',
         'objective': 'binary',
         'is_unbalance': True,
         'max_depth': -1,
         'n_estimators': 50,
         'learning_rate': 0.05,
         "min_child_samples": 30,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_fraction": 0.9 ,
         "lambda_l1": 0.1,
         "verbosity": -1}

params = {'num_leaves': 120,
          'min_data_in_leaf': 30,
          'objective': 'multiclass',
          'num_class': 27,
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
#
def train_lgb(df_train, target, df_test, features, params):
    # define folds for cross validation
    folds = KFold(n_splits=10, random_state=820)
    oof = np.zeros(len(df_train))
    predictions = np.zeros(len(df_test))
    features_importance = pd.DataFrame({'Feature': [], 'Importance': []})
    for fold, (trn_idx, val_idx) in enumerate(folds.split(df_train, target)):
        print(" fold nb:{}".format(fold))
        train_df = lgb.Dataset(df_train.iloc[trn_idx][features], label=target[trn_idx])
        validation_df = lgb.Dataset(data=df_train.iloc[val_idx][features], label=target[val_idx])

        num_round = 50  # you might change this number
        clf = lgb.train(params, train_df, num_round, valid_sets=[train_df, validation_df], verbose_eval=10,
                        early_stopping_rounds=10)
        # clf = lgb.train(train_set=train_df,
        #                  params=params,
        #                  num_boost_round=optimum_boost_rounds
        #                  )
        oof[val_idx] = clf.predict(df_train.iloc[val_idx][features], num_iteration=clf.best_iteration)

        fold_importance_df = pd.DataFrame({'Feature': [], 'Importance': []})
        fold_importance_df['Feature'] = features
        fold_importance_df['Importance'] = clf.feature_importance()
        fold_importance_df["fold"] = fold + 1
        features_importance = pd.concat([features_importance, fold_importance_df], axis=0)

        predictions += clf.predict(df_test[features], num_iteration=clf.best_iteration) / folds.n_splits

    return clf, predictions, features_importance, oof

features = X_train.columns
clf, predictions, features_importance, oof= train_lgb(X_train, y_train.values ,X_test,features, param)
print(predictions)
#
# import seaborn as sns
# from matplotlib import pyplot as plt
# def display_importances(feature_importance_df_):
#     cols = feature_importance_df_[["Feature", "Importance"]].groupby("Feature").mean().sort_values(by="Importance", ascending=False)[:10].index
#     best_features = feature_importance_df_[["Feature", "Importance"]].groupby("Feature").mean().sort_values(by="Importance", ascending=False)[:50]
#     best_features.reset_index(inplace=True)
#     print(best_features.dtypes)
#     plt.figure(figsize=(8, 10))
#     sns.barplot(x="Importance", y="Feature", data=best_features)
#     plt.title('LightGBM Features (avg over folds)')
#     plt.tight_layout()
#
# display_importances(features_importance)
#
# sample_submission['isFraud'] = predictions
# sample_submission.to_csv('submission.csv')