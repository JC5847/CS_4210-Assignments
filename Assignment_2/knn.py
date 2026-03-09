#-------------------------------------------------------------------------
# AUTHOR: Julian Collado
# FILENAME: knn.py
# SPECIFICATION: KNN program for question 3e
# FOR: CS 4210- Assignment #2
# TIME SPENT: 45 minutes
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#Importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

#Reading the data in a csv file using pandas
label_encoder = LabelEncoder()

db = []
df = pd.read_csv('email_classification.csv')
for _, row in df.iterrows():
    db.append(row.tolist())

true_label = 0
false_label = 0

#Loop your data to allow each instance to be your test set
for index, i in enumerate(db):

    #Add the training features to the 20D array X removing the instance that will be used for testing in this iteration.
    #For instance, X = [[1, 2, 3, 4, 5, ..., 20]].
    #Convert each feature value to float to avoid warning messages
    #--> add your Python code here
    db_no_instance = db.copy()
    db_no_instance.pop(index)

    db_array = np.array(db_no_instance)
    X = db_array[:, :-1].astype(float).tolist()

    #Transform the original training classes to numbers and add them to the vector Y.
    #Do not forget to remove the instance that will be used for testing in this iteration.
    #For instance, Y = [1, 2, ,...].
    #Convert each feature value to float to avoid warning messages
    #--> add your Python code here
    Y = label_encoder.fit_transform(db_array[:, -1]).tolist()

    #Store the test sample of this iteration in the vector testSample
    #--> add your Python code here
    encoded_test_class = label_encoder.transform([i[-1]]).tolist()[0]
    testSample = [float(item) for item in i[:-1]]

    #Fitting the knn to the data using k = 1 and Euclidean distance (L2 norm)
    #--> add your Python code here
    clf = KNeighborsClassifier(n_neighbors = 1, metric = 'euclidean').fit(X, Y)

    #Use your test sample in this iteration to make the class prediction. For instance:
    #class_predicted = clf.predict([[1, 2, 3, 4, 5, ..., 20]])[0]
    #--> add your Python code here
    class_predicted = clf.predict([testSample])[0]

    #Compare the prediction with the true label of the test instance to start calculating the error rate.
    #--> add your Python code here
    if class_predicted == encoded_test_class:
        true_label += 1
    else:
        false_label += 1

#Print the error rate
#--> add your Python code here
error_rate = false_label / (false_label + true_label)
print(f"Error Rate: {error_rate}")






