import pandas as pd
import seaborn as sb
import sklearn as sk
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from scipy.optimize import curve_fit
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import *
from sklearn.preprocessing import *
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error

def main():
    #define the files 
    df = pd.read_csv('train.csv') 
    testdf = pd.read_csv('test.csv')

    # define all columns wnat to use from file and create train file 
    x = df.iloc[:, lambda df: [2,3,4,7,8,10,13,14,17,21,22,23,25,26,29,33]].values
    y = df.iloc[:, lambda df: [38]].values

    list1=[]
    model = LinearRegression() # what model do you want to use 

    predictions = cross_val_predict(model, x, y, cv = 10) # create folds for data to test and train and store
    #predictions from test set 
    plt.scatter(y, predictions) # create scatter plot for actual y and predictions
    table=y,predictions

    from sklearn.metrics import mean_squared_error
    np.sqrt(mean_squared_error(predictions,y))

    coeff = metrics.r2_score(y, predictions) # compare actual and predictions to get R^2 
    print(coeff)
    print(np.sqrt(mean_squared_error(predictions,y)))


    # find outliers
    for value in predictions:
      if value > 500000:
        list1.append(value)
    print(list1)
    print(table)

main()
