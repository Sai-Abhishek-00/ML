import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

data = pd.read_csv("student/student-mat.csv", sep=";")

features = data[["G1", "G2", "studytime", "failures", "absences"]]

labels = "G3"

X = np.array(features) # Features
y = np.array(data[labels]) # Labels

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

model = linear_model.LinearRegression()
model.fit(x_train, y_train)
accuracy = model.score(x_test, y_test)
print(accuracy)

print('Coefficient: \n', model.coef_) # These are each slope value
print('Intercept: \n', model.intercept_) # This is the intercept
predictions = model.predict(x_test) # Gets a list of all predictions

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])