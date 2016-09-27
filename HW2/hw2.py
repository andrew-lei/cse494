#! /usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as sm
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# Read training data
train = pd.read_csv('wine.trainingdata.txt')
# First column is label (i.e., y), rest go to X
(y_train, X_train) = (train.ix[:,0], train.ix[:,1:])

# Question 1
# Train decision tree classifier using Gini impurity
clf_gini = DecisionTreeClassifier(criterion = 'gini')
clf_gini.fit(X_train, y_train)
# Print training score for tree with Gini criterion
print 'Gini criterion training score: ', clf_gini.score(X_train, y_train)
# Save tree
tree.export_graphviz(clf_gini, out_file='gini_tree.dot')

# Question 2
# Train decision tree classifier using entropy
clf_entropy = DecisionTreeClassifier(criterion = 'entropy')
clf_entropy.fit(X_train, y_train)
# Print training score for tree with entropy
print 'Entropy criterion training score: ', clf_entropy.score(X_train, y_train)
# Save tree
tree.export_graphviz(clf_entropy, out_file='entropy_tree.dot')

# Question 3
# Read testing data
test = pd.read_csv('wine.TestData.txt')
(y_test, X_test) = (test.ix[:,0], test.ix[:,1:])
# Print training scores
print 'Gini criterion testing score: ', clf_gini.score(X_test, y_test)
print 'Entropy criterion testing score: ', clf_entropy.score(X_test, y_test)
