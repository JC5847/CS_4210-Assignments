#-------------------------------------------------------------------------
# AUTHOR: Julian Collado
# FILENAME: decision_tree.py
# SPECIFICATION: python program that inputs data from a csv file and outputs a depth-2 decision tree
# FOR: CS 4210- Assignment #1
# TIME SPENT: 2h
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import csv
db = []
X = []
Y = []

#reading the data in a csv file
with open('contact_lens.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)

#encode the original categorical training features into numbers and add to the 4D array X.
#--> add your Python code here
db_num = []

for col in zip(*db):
  unique_features = []
  db_col = []

  for item in col:
    if item not in unique_features:
      unique_features.append(item)
    db_col.append(unique_features.index(item)+1)
  db_num.append(db_col)

X = db_num[:-1]
X = [list(row) for row in zip(*X)]

#encode the original categorical training classes into numbers and add to the vector Y.
#--> add your Python code here
Y = db_num[-1]

#fitting the depth-2 decision tree to the data using entropy as your impurity measure
#--> add your Python code here
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=2).fit(X, Y)

#plotting decision tree
tree.plot_tree(clf, feature_names=['Age', 'Spectacle', 'Astigmatism', 'Tear'], class_names=['No','Yes'], filled=True, rounded=True)
plt.show()

