#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TITANIC 1

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


data_train = pd.read_csv('../../titanic_train.csv')
data_test = pd.read_csv('../../titanic_test.csv')

print(data_train.head(3))
print(data_train.sample(3))
print(data_train.tail(3))


sns.barplot(x="Embarked", y="Survived", hue="Sex", data=data_train);

plt.figure()
sns.pointplot(x="Pclass", y="Survived", hue="Sex", data=data_train,
              palette={"male": "blue", "female": "pink"},
              markers=["*", "o"], linestyles=["-", "--"]);

def simplify_ages(df):
    df.Age = df.Age.fillna(-0.5)
    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)
    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
    categories = pd.cut(df.Age, bins, labels=group_names)
    df.Age = categories
    return df

def simplify_cabins(df):
    df.Cabin = df.Cabin.fillna('N')
    df.Cabin = df.Cabin.apply(lambda x: x[0])
    return df

def simplify_fares(df):
    df.Fare = df.Fare.fillna(-0.5)
    bins = (-1, 0, 8, 15, 31, 1000)
    group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']
    categories = pd.cut(df.Fare, bins, labels=group_names)
    df.Fare = categories
    return df

def format_name(df):
    df['Lname'] = df.Name.apply(lambda x: x.split(' ')[0])
    df['NamePrefix'] = df.Name.apply(lambda x: x.split(' ')[1])
    return df

def drop_features(df):
    return df.drop(['Ticket', 'Name', 'Embarked'], axis=1)

def transform_features(df):
    df = simplify_ages(df)
    df = simplify_cabins(df)
    df = simplify_fares(df)
    df = format_name(df)
    df = drop_features(df)
    return df

data_train = transform_features(data_train)
data_test = transform_features(data_test)
data_train.head()

#plt.figure()
#sns.barplot(x="Age", y="Survived", hue="Sex", data=data_train);
#
#plt.figure()
#sns.barplot(x="Cabin", y="Survived", hue="Sex", data=data_train);
#
#plt.figure()
#sns.barplot(x="Fare", y="Survived", hue="Sex", data=data_train);
