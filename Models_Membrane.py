# -*- coding: utf-8 -*-
"""
Spyder Editor

"""
import pandas as pd
#import numpy as np

df = pd.read_csv('R_jv.csv')
df = df.drop(['Run'],axis='columns')

df_r = df['R']
df_jv = df['Jv']

parameters_df = df.drop(['Jv','R'],axis='columns')

''' Model for predcition of %R'''

from sklearn.model_selection import train_test_split

p1_train, p1_test, df_r_train, df_r_test = train_test_split(parameters_df, df_r, test_size=0.2)
p2_train, p2_test, df_jv_train, df_jv_test = train_test_split(parameters_df, df_jv, test_size=0.2)
#print(p1_train.shape, p1_test.shape, df_r_train.shape, df_r_test.shape)

''' 
We will be implementing many models and performace metric for the
ressression algorithms will be MAPE(Mean absolute percentage error) 
'''
from sklearn.metrics import mean_absolute_percentage_error

''' SVR '''
from sklearn.svm import SVR
svr_r_rbf = SVR(kernel='rbf', gamma='auto')
svr_jv_rbf = SVR(kernel='rbf')
svr_r_linear = SVR(kernel='linear', gamma='auto')
svr_jv_linear = SVR(kernel='linear')
svr_r_poly = SVR(kernel='poly', gamma='auto')
svr_jv_poly = SVR(kernel='poly')

'''SVR rbf'''
print(" ")
print("SVR rbf")
svr_r_rbf.fit(p1_train, df_r_train)
pred_svr_r_rbf = svr_r_rbf.predict(p1_test) 

svr_jv_rbf.fit(p2_train, df_jv_train)
pred_svr_jv_rbf = svr_jv_rbf.predict(p2_test)

mape_svr_r_rbf = (mean_absolute_percentage_error(df_r_test, pred_svr_r_rbf))*100
mape_svr_jv_rbf = (mean_absolute_percentage_error(df_jv_test, pred_svr_jv_rbf))*100

print("Mean absolute percentage error for rbf support vector regression for %R is", mape_svr_r_rbf)
print("Mean absolute percentage error for rbf support vector regression for Jv is", mape_svr_jv_rbf)
print("______________________________________________________________________")

'''SVR linear'''
print(" ")
print("SVR Linear")
svr_r_linear.fit(p1_train, df_r_train)
pred_svr_r_linear = svr_r_linear.predict(p1_test) 

svr_jv_linear.fit(p2_train, df_jv_train)
pred_svr_jv_linear = svr_jv_linear.predict(p2_test)

mape_svr_r_linear = (mean_absolute_percentage_error(df_r_test, pred_svr_r_linear))*100
mape_svr_jv_linear = (mean_absolute_percentage_error(df_jv_test, pred_svr_jv_linear))*100

print("Mean absolute percentage error for linear Support Vector Regression for %R is", mape_svr_r_linear)
print("Mean absolute percentage error for linear Support Vector Regression for Jv is", mape_svr_jv_linear)
print("______________________________________________________________________")

'''SVR poly'''
print(" ")
print("SVR Polynomial")
svr_r_poly.fit(p1_train, df_r_train)
pred_svr_r_poly = svr_r_poly.predict(p1_test)

svr_jv_poly.fit(p2_train, df_jv_train)
pred_svr_jv_poly = svr_jv_poly.predict(p2_test)

mape_svr_r_poly = (mean_absolute_percentage_error(df_r_test, pred_svr_r_poly))*100
mape_svr_jv_poly = (mean_absolute_percentage_error(df_jv_test, pred_svr_jv_poly))*100

print("Mean absolute percentage error for polynomial support vector regression for %R is", mape_svr_r_poly)
print("Mean absolute percentage error for polynomial support vector regression for Jv is", mape_svr_jv_poly)
print("______________________________________________________________________")

'''SGD'''
print(" ")
print("Stochastic Gradient Descent")
from sklearn.linear_model import SGDRegressor
sgd_r = SGDRegressor(max_iter=3, loss='huber') #huber loss cost function
sgd_r.fit(p1_train, df_r_train)
pred_sdg_r = sgd_r.predict(p1_test) 

sgd_jv = SGDRegressor(max_iter=3, loss='huber') #huber loss cost function
sgd_jv.fit(p2_train, df_jv_train)
pred_sdg_jv = sgd_jv.predict(p2_test) 

mape_sgd_r = (mean_absolute_percentage_error(df_r_test, pred_sdg_r))*100
mape_sgd_jv = (mean_absolute_percentage_error(df_jv_test, pred_svr_jv_poly))*100

print("Mean absolute percentage error for stochastic gradient descent for %R is", mape_sgd_r)
print("Mean absolute percentage error for stochastic gradient descent for Jv is", mape_sgd_jv)
print("______________________________________________________________________")

'''PLS'''
print(" ")
print("Partial Least Square")
from sklearn.cross_decomposition import PLSRegression
pls_r = PLSRegression(n_components=4, max_iter=5)
pls_r.fit(p1_train, df_r_train)
pred_pls_r = pls_r.predict(p1_test)

pls_jv = PLSRegression(n_components=4, max_iter=5)
pls_jv.fit(p2_train, df_jv_train)
pred_pls_jv = pls_jv.predict(p2_test)


mape_pls_r = (mean_absolute_percentage_error(df_r_test, pred_sdg_r))*100
mape_pls_jv = (mean_absolute_percentage_error(df_jv_test, pred_sdg_jv))*100

print("Mean absolute percentage error for Partial Least Square Regression for %R is", mape_pls_r)
print("Mean absolute percentage error for Partial Least Square Regression for Jv is", mape_pls_jv)
print("______________________________________________________________________")

'''Decision Tree Regressor'''
print(" ")
print("Decision Tree Regressor")
from sklearn.tree import DecisionTreeRegressor
dtr_r = DecisionTreeRegressor(max_depth=4)
dtr_r.fit(p1_train, df_r_train)
pred_dtr_r = dtr_r.predict(p1_test)

dtr_jv = DecisionTreeRegressor(max_depth=4)
dtr_jv.fit(p2_train, df_jv_train)
pred_dtr_jv = dtr_jv.predict(p2_test)

mape_dtr_r = (mean_absolute_percentage_error(df_r_test, pred_dtr_r))*100
mape_dtr_jv = (mean_absolute_percentage_error(df_jv_test, pred_dtr_jv))*100

print("Mean absolute percentage error for Decision Tree Regression for %R is", mape_dtr_r)
print("Mean absolute percentage error for Decision Tree Regression for Jv is", mape_dtr_jv)
print("______________________________________________________________________")

''' Elastic Net'''
print(" ")
print("Elastic Net")
from sklearn.linear_model import ElasticNet
en_r = ElasticNet()
en_r.fit(p1_train, df_r_train)
pred_en_r = en_r.predict(p1_test)

en_jv = ElasticNet()
en_jv.fit(p2_train, df_jv_train)
pred_en_jv = en_jv.predict(p2_test)

mape_en_r = (mean_absolute_percentage_error(df_r_test, pred_en_r))*100
mape_en_jv = (mean_absolute_percentage_error(df_jv_test, pred_en_jv))*100

print("Mean absolute percentage error for ElasticNet for %R is", mape_en_r)
print("Mean absolute percentage error for ElasticNet for Jv is", mape_en_jv)
print("______________________________________________________________________")

'''Bayesian Ridge'''
print(" ")
print("Bayesian Ridge")
from sklearn.linear_model import BayesianRidge
br_r = BayesianRidge()
br_r.fit(p1_train, df_r_train)
pred_br_r = br_r.predict(p1_test)

br_jv = BayesianRidge()
br_jv.fit(p2_train, df_jv_train)
pred_br_jv = br_jv.predict(p2_test)

mape_br_r = (mean_absolute_percentage_error(df_r_test, pred_br_r))*100
mape_br_jv = (mean_absolute_percentage_error(df_jv_test, pred_br_jv))*100

print("Mean absolute percentage error for Bayesian Ridge for %R is", mape_br_r)
print("Mean absolute percentage error for Bayesian Ridge for Jv is", mape_br_jv)
print("______________________________________________________________________")

'''ARDRegression, Automatic Relevance Determinantion Regression'''
print(" ")
print("Automatic Relevance Determinantion Regression")
from sklearn.linear_model import ARDRegression
ard_r = ARDRegression()
ard_r.fit(p1_train, df_r_train)
pred_ard_r = ard_r.predict(p1_test)

ard_jv = ARDRegression()
ard_jv.fit(p2_train, df_jv_train)
pred_ard_jv = ard_jv.predict(p2_test)

mape_ard_r = (mean_absolute_percentage_error(df_r_test, pred_ard_r))*100
mape_ard_jv = (mean_absolute_percentage_error(df_jv_test, pred_ard_jv))*100

print("Mean absolute percentage error for ARDRegression for %R is", mape_ard_r)
print("Mean absolute percentage error for ARDRegression for Jv is", mape_ard_jv)
print("______________________________________________________________________")

'''TweedieRegressor, Power 1: Poisson, Power 2: Gamma, Power 3: Inverse Gaussian'''
print(" ")
from sklearn.linear_model import TweedieRegressor
print(" Tweedie Regressor - Poisson and Gamma")
trpg_r = TweedieRegressor(power=1.5, link='log')
trpg_r.fit(p1_train, df_r_train)
pred_trpg_r = trpg_r.predict(p1_test)

trpg_jv = TweedieRegressor(power=1.5, link='log')
trpg_jv.fit(p2_train, df_jv_train)
pred_trpg_jv = trpg_jv.predict(p2_test)

mape_trpg_r = (mean_absolute_percentage_error(df_r_test, pred_trpg_r))*100
mape_trpg_jv = (mean_absolute_percentage_error(df_jv_test, pred_trpg_jv))*100

print("Mean absolute percentage error for combined poisson gamma tweedie regressor for %R is", mape_trpg_r)
print("Mean absolute percentage error for combined poisson gamma tweedie regressor for Jv is", mape_trpg_jv)
print("______________________________________________________________________")
print(" ")
print(" Tweedie Regressor - Poisson")
trp_r = TweedieRegressor(power=1, link='log')
trp_r.fit(p1_train, df_r_train)
pred_trp_r = trp_r.predict(p1_test)

trp_jv = TweedieRegressor(power=1, link='log')
trp_jv.fit(p2_train, df_jv_train)
pred_trp_jv = trp_jv.predict(p2_test)

mape_trp_r = (mean_absolute_percentage_error(df_r_test, pred_trp_r))*100
mape_trp_jv = (mean_absolute_percentage_error(df_jv_test, pred_trp_jv))*100

print("Mean absolute percenatge error for poisson tweedie regressor for %R is", mape_trp_r)
print("Mean absolute percentage error for poisson tweedie regressor for Jv is", mape_trp_jv)
print("______________________________________________________________________")
print(" ")
print(" Tweedie Regressor - Gamma")
trg_r = TweedieRegressor(power=2, link='log')
trg_r.fit(p1_train, df_r_train)
pred_trg_r = trg_r.predict(p1_test)

trg_jv = TweedieRegressor(power=1, link='log')
trg_jv.fit(p2_train, df_jv_train)
pred_trg_jv = trg_jv.predict(p2_test)

mape_trg_r = (mean_absolute_percentage_error(df_r_test, pred_trg_r))*100
mape_trg_jv = (mean_absolute_percentage_error(df_jv_test, pred_trg_jv))*100

print("Mean absolute percentage error for gamma tweedie regressor for %R is", mape_trg_r)
print("Mean absolute percentage error for gamma tweedie regressor for Jv is", mape_trg_jv)
print("______________________________________________________________________")
''' K Nearest Neighbours Regressor'''
print(" ")
print("K Nearest Neighbour Regressor")
from sklearn.neighbors import KNeighborsRegressor
knn_r = KNeighborsRegressor()
knn_r.fit(p1_train, df_r_train)
pred_knn_r = knn_r.predict(p1_test)

knn_jv = KNeighborsRegressor()
knn_jv.fit(p2_train, df_jv_train)
pred_knn_jv = knn_jv.predict(p2_test)

mape_knn_r = (mean_absolute_percentage_error(df_r_test, pred_knn_r))*100
mape_knn_jv = (mean_absolute_percentage_error(df_jv_test, pred_knn_jv))*100

print("Mean absolute percentage error for KNN regressor for %R is", mape_knn_r)
print("Mean absolute percentage error for KNN regressor for Jv is", mape_knn_jv)
print(" ")
print(" ")

L_r = [mape_svr_r_rbf, mape_svr_r_poly, mape_svr_r_linear, mape_knn_r, mape_trg_r, mape_trp_r, mape_trpg_r, mape_ard_r, mape_br_r, mape_en_r, mape_dtr_r, mape_pls_r, mape_sgd_r]
L_jv = [mape_svr_jv_rbf, mape_svr_jv_poly, mape_svr_jv_linear, mape_knn_jv, mape_trg_jv, mape_trp_jv, mape_trpg_jv, mape_ard_jv, mape_br_jv, mape_en_jv, mape_dtr_jv, mape_pls_jv, mape_sgd_jv]
L = ['SVR rbf', 'SVR Polynomial', 'SVR Linear', 'K-Nearest Neighnour', 'Tweedie - Gamma','Tweedie - Poisson', 'Tweedie - Poisson & Gamma', 'ARDRegression', 'Bayesian Ridge', 'Elastic Net', 'Decision Tree', 'Partial Least Squares', 'Stochastic Gradient Descent']
Result = pd.DataFrame({'Model': L, '%R MAPE': L_r, 'Jv MAPE': L_jv})
Result
Result.to_excel("Output5.xlsx")



