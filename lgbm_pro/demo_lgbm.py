import pandas as pd
import numpy as np
from sklearn.cross_validation import KFold
import re
# import plotly.graph_objs as go
# import plotly.offline as py
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC
import lightgbm as lgbm
import warnings
warnings.filterwarnings('ignore')  # 忽略warning
pd.set_option('display.max_columns', None)  # 输出结果显示全部列

train = pd.read_csv('./titanic_data/train.csv')
test = pd.read_csv('./titanic_data/test.csv')
PassengerId = test['PassengerId']
full_data = [train, test]

# 查看train集的数据
# print(train.describe())  # 查看描述性统计,只能看数值型数据。
# print(train.info())  # 查看数据的信息
# print(train.head())  # 查看train的前n行数据，默认为前5行

# 添加新的特征，名字的长度
train['Name_length'] = train['Name'].apply(len)
test['Name_length'] = test['Name'].apply(len)

# 乘客在船上是否有船舱
train['Has_Cabin'] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
test['Has_Cabin'] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

# 结合SibSp和Parch创建新的特性FamilySize
for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

# 定义从乘客名中提取新的特征[Title]的函数
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # 如果title存在，提取并返回它。
    if title_search:
        return title_search.group(1)
    return ""


# 创建一个新的特征[Title]
for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)
# 将所有不常见的Title分组为一个“Rare”组
for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(
        ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')


# 通过统计三个登船地点人数最多的填充缺失值
for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')

# 缺失值填充，Test集的Fare有一个缺失，按中位数来填充,以及创建一个新的特征[CategoricalFare]
for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
train['CategoricalFare'] = pd.qcut(train['Fare'], 4)

# 缺失值填充,以及创建新的特征[CategoricalAge]
for dataset in full_data:
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)


train['CategoricalAge'] = pd.cut(train['Age'], 5)
# print(train['CategoricalAge'])


for dataset in full_data:
    dataset['Sex'] = dataset['Sex'].map({'female': 0, 'male': 1}).astype(int)

    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[dataset['Age'] > 64, 'Age'] = 4


drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']
train = train.drop(drop_elements, axis=1)
train = train.drop(['CategoricalAge', 'CategoricalFare'], axis=1)
test = test.drop(drop_elements, axis=1)
# print(train.head())
# print(train.describe())
# print(train.head())


#模型方面
# 一些有用的参数，下面会用到
ntrain = train.shape[0]
ntest = test.shape[0]
SEED = 0
NFOLDS = 5
kf = KFold(ntrain, n_folds=NFOLDS, random_state=SEED)

class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)

    def fit(self, x, y):
        return self.clf.fit(x, y)

    def feature_importances(self, x, y):
        return self.clf.fit(x, y).feature_importances_

def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

# 随机森林的参数
rf_params = {
    'n_jobs': -1,
    'n_estimators': 100,
     'warm_start': True,
     #'max_features': 0.2,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
    'verbose': 0
}

# Extra Trees的参数
et_params = {
    'n_jobs': -1,
    'n_estimators': 100,
    #'max_features': 0.5,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0
}

# AdaBoost的参数
ada_params = {
    'n_estimators': 100,
    'learning_rate': 0.01
}

# Gradient Boosting的参数
gb_params = {
    'n_estimators': 100,
     #'max_features': 0.2,
    'max_depth': 5,
    'min_samples_leaf': 2,
    'verbose': 0
}

# Support Vector Classifier的参数
svc_params = {
    'kernel': 'linear',
    'C': 0.025
}

# 通过前面定义的SklearnHelper类创建5个对象来表示5个学习模型
rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)
# 创建包含train、test的Numpy数组，以提供给我们的模型
y_train = train['Survived'].ravel()
train = train.drop(['Survived'], axis=1)
x_train = train.values
# test = test.drop(['Parch', 'Embarked', 'Has_Cabin', 'IsAlone'], axis=1)
x_test = test.values

#这些将会作为新的特征被使用
et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test)  # Extra Trees
rf_oof_train, rf_oof_test = get_oof(rf, x_train, y_train, x_test)  # Random Forest
ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test)  # AdaBoost
gb_oof_train, gb_oof_test = get_oof(gb, x_train, y_train, x_test)  # Gradient Boost
svc_oof_train, svc_oof_test = get_oof(svc, x_train, y_train, x_test)  # Support Vector Classifier

rf_features = rf.feature_importances(x_train, y_train)
et_features = et.feature_importances(x_train, y_train)
ada_features = ada.feature_importances(x_train, y_train)
gb_features = gb.feature_importances(x_train, y_train)

cols = train.columns.values
feature_dataframe = pd.DataFrame({'features': cols,
     'Random Forest feature importances': rf_features,
     'Extra Trees  feature importances': et_features,
      'AdaBoost feature importances': ada_features,
    'Gradient Boost feature importances': gb_features})

feature_dataframe['mean'] = feature_dataframe.mean(axis=1)  # axis = 1 computes the mean row-wise
# print(feature_dataframe.head(11))

x_train = np.concatenate((et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train), axis=1)
x_test = np.concatenate((et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test), axis=1)


lgbm_train = lgbm.Dataset ( data=x_train ,
                            label=y_train)
lgbm_params = {
    'boosting': 'dart' ,
    'application': 'binary' ,
    'learning_rate': 0.01 ,
    'feature_fraction': 0.5 ,
    'verbose' : -1,
    'drop_rate': 0.02
}

cv_results = lgbm.cv ( train_set=lgbm_train ,
                       params=lgbm_params ,
                       nfold=5 ,
                       num_boost_round=600 ,
                       early_stopping_rounds=50 ,
                       verbose_eval=50 ,

                       metrics=['auc'] )

optimum_boost_rounds = np.argmax ( cv_results['auc-mean'] )
print ( 'Optimum boost rounds = {}'.format ( optimum_boost_rounds ) )
print ( 'Best CV result = {}'.format ( np.max ( cv_results['auc-mean'] ) ) )

clf = lgbm.train ( train_set=lgbm_train ,
                   params=lgbm_params ,
                   num_boost_round=optimum_boost_rounds
)
##预测结果为浮点数，而本次比赛的预测结果需要0，1，所以将其转换
predictions = clf.predict ( x_test )
predictions = predictions + 0.5
predictions = predictions.astype(int)


