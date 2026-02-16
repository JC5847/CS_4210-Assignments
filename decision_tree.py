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
mapping = {'Young': 0, 'Presbyopic': 1, 'Prepresbyopic': 2, 'Myope': 0, 'Hypermetrope': 1, 'No': 1, 'Yes': 1, 'Normal': 0, 'Reduced': 1}
X = [mapping[item] for item in db]
print(X)

#encode the original categorical training classes into numbers and add to the vector Y.
#--> addd your Python code here
# Y =

#fitting the depth-2 decision tree to the data using entropy as your impurity measure
#--> addd your Python code here
#clf =

#plotting decision tree
tree.plot_tree(clf, feature_names=['Age', 'Spectacle', 'Astigmatism', 'Tear'], class_names=['Yes','No'], filled=True, rounded=True)
plt.show()