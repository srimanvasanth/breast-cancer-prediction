#Importing necessary libraries
import numpy as np
import sklearn.datasets
import pandas as pd

#Loading breast cancer dataset
breast_cancer = sklearn.datasets.load_breast_cancer()

#Declaring variables
X = breast_cancer.data
Y = breast_cancer.target

#Importing data to panda dataframe
data = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
data['class'] = breast_cancer.target
data.head()

#Train-Test split
from sklearn.model_selection import train_test_split
(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, random_state=1, test_size=0.15)

#Linear Regression model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(solver='lbfgs', max_iter=4000)
classifier.fit(X, Y)

#Accuracy score for accuracy checking
from sklearn.metrics import accuracy_score
acc_train = accuracy_score(Y_train, classifier.predict(X_train))
print(acc_train)
acc_test = accuracy_score(Y_test, classifier.predict(X_test))
print(acc_test)

#Detecting whether the patient has breast cancer
input_data1 = (13.08, 15.71,	85.63, 520,	0.1075,	0.127, 0.04568,	0.0311,	0.1967,	0.06811, 0.1852, 0.7477, 1.383,	14.67, 0.004097, 0.01898, 0.01698, 0.00649,	0.01678, 0.002425, 14.5, 20.49,	96.09, 630.5, 0.1312, 0.2776, 0.189, 0.07283, 0.3184, 0.08183)
input_data2 = (13.28, 20.28,	87.32,	545.2,	0.1041,	0.1436,	0.09847,	0.06158,	0.1974,	0.06782,	0.3704,	0.8249,	2.427,	31.33,	0.005072,	0.02147,	0.02185,	0.00956,	0.01719,	0.003317,	17.38,	28,	113.1,	907.2,	0.153,	0.3724,	0.3664,	0.1492,	0.3739,	0.1027)

input_array1 = np.array(input_data1)
input_array2 = np.array(input_data2)

test_data1 = input_array1.reshape(1, -1)
test_data2 = input_array2.reshape(1, -1)

#Predicting the output
pred1 = classifier.predict(test_data1)
pred2 = classifier.predict(test_data2)

if pred1[0] == 0:
    print("Cancer is Malignant")
else:
    print("Cancer is Benign")

if pred2[0] == 0:
    print("Cancer is Malignant")
else:
    print("Cancer is Benign")




