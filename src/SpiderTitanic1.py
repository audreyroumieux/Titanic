# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 09:39:02 2018
@author: audrey roumieux
Projet: Titanic
Description:
"""
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn import linear_model

train = pd.read_csv('titanic_train.csv')
test = pd.read_csv('titanic_test.csv')


columns = ['PassengerId','Name', 'Ticket', 'Cabin']
train.drop(columns, inplace=True, axis=1) #suprimer de colonnes inninteressantes
test.drop(columns, inplace=True, axis=1)


train.replace(to_replace=dict(female=1, male=0, Q=2, S=1, C=0), inplace=True)
test.replace(to_replace=dict(female=1, male=0, Q=2, S=1, C=0), inplace=True)
#test.replace(to_replace=dict(Q=2, S=1, C=0), inplace=True)

#supression des ligne avec nan dans embarked
#index_nan=train[(train['Embarked'] != 'S')&(train['Embarked'] != 'C')&(train['Embarked'] != 'Q')].index
#train.drop(index_nan, inplace=True)
train["Embarked"] = train["Embarked"].fillna(train["Embarked"].median())
train["Age"] = train["Age"].fillna(train["Age"].median()) #modification des nan par median dans age

test["Age"] = test["Age"].fillna(test["Age"].median())
test["Embarked"] = test["Embarked"].fillna(test["Embarked"].median())
test["Fare"] = test["Fare"].fillna(test["Fare"].median())


x_train, x_test, y_train, y_test = train_test_split(train.drop("Survived", axis=1), train.Survived, test_size=.3, random_state=42)


#plt.figure(1, figsize=(9, 6))
##plt.boxplot(train[["Pclass", "Survived"]])
#plt.boxplot(x_train, y_train)
#plt.show()


regL=linear_model.LogisticRegression()
regL.fit(x_train, y_train)
regL.predict(x_test)
print(regL.score(x_train, y_train))
"""
"""

from sklearn import tree
regT=tree.DecisionTreeClassifier()
regT.fit(x_train, y_train)
regT.score(x_train, y_train)

import pydotplus 
import collections

dot_data = tree.export_graphviz(regT, out_file=None,
                        feature_names=x_train.columns,  
                         filled=True, rounded=True)  
graph = pydotplus.graph_from_dot_data(dot_data) 
colors=('turquoise', 'orange')
edges = collections.defaultdict(list)

for edge in graph.get_edge_list():
    edges[edge.get_source()].append(int(edge.get_destination()))
    
for edge in edges:
    edges[edge].sort()
    for i in range():
        dest= graph.get_node(str(edges[edge][i]))[0]
        dest.set_fillcolor(colors[i])
graph  
