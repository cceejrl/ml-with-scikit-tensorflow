#!/usr/bin/python
# -*- coding: utf-8 -*-
# encoding: utf-8

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import hashlib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.tools.plotting import scatter_matrix
# import pandas.plotting.scatter_matrix
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion

from myCustomTrans import CombinedAttributesAdder, DataFrameSelector

HOUSING_PATH = "../../datasets/housing"


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    # print(csv_path)
    return pd.read_csv(csv_path)


housing = load_housing_data()
# print(housing.info())
# print(housing["ocean_proximity"].value_counts())
# print(housing.describe())

# housing.hist(bins=50, figsize=(20, 15))
# plt.show()


def split_train_test(data, test_ratio):
    np.random.seed(55)
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


# create a test set
train_set, test_set = split_train_test(housing, 0.2)
# print("train_set.size() = ", len(train_set), ", test_set.size() = ", len(test_set))


def test_set_check(identifier, test_ratio, hash):
    # print(np.int64(identifier))
    # print(hash(np.int64(identifier)).digest()[-1])
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio


def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]


housing_with_id = housing.reset_index() # adds an `index` column
# train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, True)
smt = housing["income_cat"].value_counts() / len(housing)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

for set in (strat_train_set, strat_test_set):
    set.drop(["income_cat"], axis=1, inplace=True)

# housing = strat_train_set.copy()
# housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
#              s=housing["population"]/100, label="population",
#              c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
#              )
# plt.legend()
# plt.show()

# attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
# scatter_matrix(housing[attributes], figsize=(12,8))

# housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.2)

housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

imputer = Imputer(strategy="median")
housing_num = housing.drop("ocean_proximity", axis=1)

imputer.fit(housing_num)

# print(imputer.statistics_)
# print(housing_num.median().values)

X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns)
# print(housing_tr)

encoder = LabelEncoder()
housing_cat = housing["ocean_proximity"]
# housing_cat_encoded = encoder.fit_transform(housing_cat)
# print(housing_cat_encoded)
# print(encoder.classes_)
# encoder = OneHotEncoder()
# housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1))
# print(housing_cat_1hot.toarray())

encoder = LabelBinarizer(sparse_output=False)
housing_cat_1hot = encoder.fit_transform(housing_cat)
# print(housing_cat_1hot)

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)
# print(housing_extra_attribs)

num_pipeline = Pipeline([('imputer', Imputer(strategy="median")),
                         ('attribs_addr', CombinedAttributesAdder()),
                         ('std_scaler', StandardScaler())])

housing_num_tr = num_pipeline.fit_transform(housing_num)
# print(housing_num_tr)

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]
num_pipeline = Pipeline([('selector', DataFrameSelector(num_attribs)),
                         ('imputer', Imputer(strategy="median")),
                         ('attribs_adder', CombinedAttributesAdder()),
                         ('std_scaler', StandardScaler())])
cat_pipline = Pipeline([('selector', DataFrameSelector(cat_attribs)),
                        ('label_binarizer', LabelBinarizer())])
full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipline)
    ])

housing_prepared = full_pipeline.fit_transform(housing)

# print(housing_prepared)
