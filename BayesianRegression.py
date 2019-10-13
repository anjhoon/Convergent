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

def main():
    # Multiple Regression using Backward Elimination and Cross_Val

    # df = pd.read_csv('train_adjusted.csv')
    df = pd.read_csv('train_adjusted.csv')
    columns = list(df.columns.values)


    # train all columns
    x = df[['LotArea',
     'OverallQual',
     'OverallCond',
     'MasVnrArea',
     'BsmtFinSF1',
     'BsmtUnfSF',
     '1stFlrSF',
     '2ndFlrSF',
     'BsmtFullBath',
     'BedroomAbvGr',
     'KitchenAbvGr',
     'TotRmsAbvGrd',
     'Fire2laces',
     'GarageYrBlt',
     'GarageCars',
     'WoodDeckSF',
     'ScreenPorch']]

    # y = df.iloc[:, lambda df: [38]].values
    y = df[['SalePrice']]

    poly = PolynomialFeatures(degree=2,include_bias=False)
    x = poly.fit_transform(x)
    sds =  StandardScaler()
    x = sds.fit_transform(x)

    model = BayesianRidge()
    model.fit(x,y)

    scores = cross_val_score(model, x, y, cv = 10)
    print(scores)

    predictions = cross_val_predict(model, x, y, cv = 10)
    plt.scatter(y, predictions)

    coeff = metrics.r2_score(y, predictions)
    print("R^2 Value:", coeff)

    rmse = np.sqrt(mean_squared_error(predictions,y))
    print('Root Mean Squared Error:', rmse)

    plt.scatter(y,predictions)
    plt.xlabel("Actual Sales Price")
    plt.ylabel("Predictions")

main()
