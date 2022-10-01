from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

api = KaggleApi()
api.authenticate()
api.dataset_download_file('rsadiq/salary', file_name = 'Salary.csv')

#Read in our dataset
df = pd.read_csv('Salary.csv')

#Converting our feature and label columns into arrays
X = np.array(df.iloc[:, 0])
X = X.reshape(-1, 1)
y = np.array(df.iloc[:, 1])

#Splitting our data into train and test subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .28)

#Finding a line of best fit for our train data
regressor = LinearRegression()
regressor.fit(X_train, y_train)
accuracy = regressor.score(X_test, y_test)
print()
print('Accuracy:', accuracy)
print()

#Visualizing our line of best fit with our train data
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show(block = False)

#Comparing our y_pred to our y_test
y_pred = regressor.predict((X_test))
y_pred = y_pred.astype(int)
comparison = []
for i in range(len(y_test)):
    a_list = []
    a_list.append(y_pred[i])
    a_list.append(y_test[i])
    comparison.append(a_list)
print('Comparison (prediction, actual):', comparison)

#We can see that our model consistently hits accuracy around 96-98%, showing a strong correlation between years of
#experience and salary in this data set.
