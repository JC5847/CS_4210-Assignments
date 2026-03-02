#-------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #2
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#Importing some Python libraries
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
import pandas as pd

import matplotlib.pyplot as plt

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']
label_encoder = LabelEncoder()

#Reading the test data in a csv file using pandas
dbTest = []
df_test = pd.read_csv('contact_lens_test.csv')
for _, row in df_test.iterrows():
    dbTest.append(row.tolist())

for ds in dataSets:

    dbTraining = []
    X = []
    Y = []

    #Reading the training data in a csv file using pandas
    # --> add your Python code here
    dfTraining = pd.read_csv(ds)
    for _, row in dfTraining.iterrows():
        dbTraining.append(row.tolist())

    #Transform the original categorical training features to numbers and add to the 4D array X.
    #For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3, X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    #--> add your Python code here
    dbTraining_num = []

    for col in zip(*dbTraining):
        dbTraining_col = label_encoder.fit_transform(col).tolist()
        dbTraining_num.append(dbTraining_col)

    X = dbTraining_num[:-1]
    X = [list(row) for row in zip(*X)]

    #Transform the original categorical training classes to numbers and add to the vector Y.
    #For instance Yes = 1 and No = 2, Y = [1, 1, 2, 2, ...]
    #--> add your Python code here
    Y = dbTraining_num[-1]

    #Loop your training and test tasks 10 times here
    accuracy = []

    for i in range (10):

       # fitting the decision tree to the data using entropy as your impurity measure and maximum depth = 5
       # --> addd your Python code here
       clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 5).fit(X, Y)

       #Read the test data and add this data to dbTest
       #--> add your Python code here
       dbTest_num = []
       for col in zip(*dbTest):
           dbTest_col=label_encoder.fit_transform(col).tolist()
           dbTest_num.append(dbTest_col)
       dbTest_new = [list(row) for row in zip(*dbTest_num)]

       true_label = 0
       false_label = 0

       for data in dbTest_new:
            #Transform the features of the test instances to numbers following the same strategy done during training,
            #and then use the decision tree to make the class prediction. For instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
            #where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
            #--> add your Python code here
            test_instance = data[:4]
            class_predicted = clf.predict([test_instance])[0]

            #Compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
            #--> add your Python code here
            if data[4] == class_predicted:
                true_label += 1
            else:
                false_label += 1

       accuracy.append(true_label / (true_label + false_label))

    #Find the average of this model during the 10 runs (training and test set)
    #--> add your Python code here
    total_accuracy = 0
    for val in accuracy:
        total_accuracy += val
    avg_accuracy = total_accuracy / 10

    #Print the average accuracy of this model during the 10 runs (training and test set).
    #Your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    #--> add your Python code here
    print(f"Final accuracy when training on {ds}: {avg_accuracy}")



