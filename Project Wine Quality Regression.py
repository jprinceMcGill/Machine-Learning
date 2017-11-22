# Wine Quality  - Regression
#P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. 
#  Modeling wine preferences by data mining from physicochemical properties.
#  In Decision Support Systems, Elsevier, 47(4):547-553. ISSN: 0167-9236.
# Version 1.2
#
#  Available at: [@Elsevier] http://dx.doi.org/10.1016/j.dss.2009.05.016
#                [Pre-press (pdf)] http://www3.dsi.uminho.pt/pcortez/winequality09.pdf
#                [bib] http://www3.dsi.uminho.pt/pcortez/dss09.bib

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset

dataset = pd.read_csv('Winequality-red.csv', sep=';')
X = dataset.iloc[:, [0]].values
y = dataset.iloc[:, 11].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Visualising the Regression results
#plt.scatter(X, y, color = 'blue')
#plt.plot(X, regressor.predict(X), color = 'red')
#plt.title('Truth or Bluff (Regression Model)')
#plt.xlabel('Characteristics')
#plt.ylabel('Quality')
#plt.show()

import statsmodels.formula.api as sm
X = np.append(arr = np.ones((1599,1)).astype(int), values = X, axis = 1)
X_opt = X
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()


# Building the optimal Model


#X_opt = X[:,[0,1,2,3,4,5,6,7,8,9,10,11]]
#regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
#regressor_OLS.summary()
#X_opt = X[:,[0,1,2,3,4,5,6,7,9,10,11]]
#regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
#regressor_OLS.summary()
#X_opt = X[:,[0,2,3,4,5,6,7,9,10,11]]
#regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
#regressor_OLS.summary()
#X_opt = X[:,[0,2,3,5,6,7,9,10,11]]
#regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
#regressor_OLS.summary()
#X_opt = X[:,[0,2,5,6,7,9,10,11]]
#regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
#regressor_OLS.summary()

