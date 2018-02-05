#!/usr/bin/python
# -*- coding: utf-8 -*-
# encoding: utf-8

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

HOUSING_PATH = "../../datasets/housing"


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    print(csv_path)
    return pd.read_csv(csv_path)


housing = load_housing_data()
print(housing.info())
print(housing["ocean_proximity"].value_counts())
print(housing.describe())

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
print("train_set.size() = ", len(train_set), ", test_set.size() = ", len(test_set))
