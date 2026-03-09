#-------------------------------------------------------------------------
# AUTHOR: Julian Collaado
# FILENAME: naive_bayes.py
# SPECIFICATION: Naive Bayes program for assignment 2
# FOR: CS 4210- Assignment #2
# TIME SPENT: 30 minutes
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#Importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn.preprocessing import LabelEncoder

dbTraining = []
dbTest = []
label_encoder = LabelEncoder()

#Reading the training data using Pandas
df = pd.read_csv('weather_training.csv')
for _, row in df.iterrows():
    dbTraining.append(row.tolist())

#Transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#--> add your Python code here
dbTraining_num = []

for col in zip(*dbTraining):
    dbTraining_col = label_encoder.fit_transform(col).tolist()
    dbTraining_num.append(dbTraining_col)

X = dbTraining_num[1:-1]
X = [list(row) for row in zip(*X)]

#Transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here
Y = dbTraining_num[-1]

#Fitting the naive bayes to the data using smoothing
#--> add your Python code here
clf = GaussianNB().fit(X, Y)

#Reading the test data using Pandas
df = pd.read_csv('weather_test.csv')
for _, row in df.iterrows():
    dbTest.append(row.tolist())

#Printing the header os the solution
#--> add your Python code here
header = list(df.columns)
spacing = []

for feature in header:
    print(feature, end = "     ")
    spacing.append(len(feature) + 5)
print("Confidence")

#Use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
#--> add your Python code here
dbTest_num = []

for col in zip(*dbTest):
    dbTest_col = label_encoder.fit_transform(col).tolist()
    dbTest_num.append(dbTest_col)

test_X = dbTest_num[1:-1]
test_X = [list(row) for row in zip(*test_X)]

valid_predictions = []

for index, row in enumerate(test_X):
    prediction = clf.predict_proba([row])[0]
    if prediction[0] >= 0.75:
        valid_predictions.append([index, "No", float(prediction[0])])
    elif prediction[1] >= 0.75:
        valid_predictions.append([index, "Yes", float(prediction[1])])

for case in valid_predictions:
    curr_case = dbTest[case[0]]

    for index in range(5):
        curr_feature = curr_case[index]
        padding = " " * (spacing[index] - len(curr_feature))
        print(curr_feature, end = padding)

    padding = " " * (spacing[5] - len(case[1]))
    print(case[1], end = padding)
    print(f"{case[2]:.2f}")









