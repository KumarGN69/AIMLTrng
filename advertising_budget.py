import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plot
import seaborn as sns
import dotenv , os
from mpl_toolkits import mplot3d

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import RFE


def plot_3d_pred_homeprice(x,y,z):
    """
        :param x: Area of the home
        :param y: Number of bedrooms
        :param z: The predicted value of the home
        :return: None - No return value
    """
    fig = plot.figure()
    ax = plot.axes(projection='3d')
    ax.scatter(x,y,z)
    ax.set_title(label='3D plot Home Price Prediction')
    ax.set_xlabel('Area',fontsize=10)
    ax.set_ylabel('BedRooms',fontsize=10)
    ax.set_zlabel('Price',fontsize=10)
    plot.savefig('3dplt_pred_output.png')

# Enable reading of environment variable
dotenv.load_dotenv()

df = pd.read_csv('./tv_ad_budgets.csv')
# print(df['Sales (units)'])
inputx = df.iloc[:,0:3]
inputy = df.iloc[:,3]
print(inputx)
print(inputy)

# Supervised learning
# split the data into train and test
input_train,input_test,output_train,output_test = train_test_split(inputx,inputy,test_size=1/4,random_state=0)
#

#------------------------------------------------------------------------------
# inputx = df.iloc[:,0:3]
# inputy = df.iloc[:,3]
# # print(inputx)
# # print(inputy)
# #

#
# # create a model
# model = LinearRegression()
#
# # train the regression model
# model.fit(input_train,output_train)
# pred_output = model.predict(input_test)
#
# # print to console the score pf the model
# print(f"Auto calculated r2_score : ", model.score(input_test,output_test))
# print(f"Manually calculated r2_score : ", r2_score(output_test,pred_output))
# print(f"Mean Sqrd Error : ", root_mean_squared_error(output_test,pred_output))
# print(f"model params coeff: { model.coef_}, intercept: {model.intercept_}")
#
# # # configure matplotlib for saving to an image file
# # matplotlib.use('Agg')
# # create an image of the plot
# # plot_3d_pred_homeprice(input_test['Area'],input_test['BedRooms'],pred_output)
#
# #create a scaler object
# scaler = StandardScaler()
#
# #scale the inout train and test data sets
# input_scaled_train = scaler.fit_transform(input_train)
# input_scaled_test = scaler.transform(input_test)
#
# # train the model with scaling
# model.fit(input_scaled_train, output_train)
# pred_output2 = model.predict(input_scaled_test)
#
# print(f"Auto calculated r2_score with scaling : ", model.score(input_scaled_test,output_test))
# print(f"Manually calculated r2_score with scaling : ", r2_score(output_test,pred_output2))
# print(f"Mean Sqrd Error with scaling : ", root_mean_squared_error(output_test,pred_output2))
# print(f"model params coeff with scaling: { model.coef_}, intercept: {model.intercept_}")
# #
# poly = PolynomialFeatures(degree=2, include_bias=False)
# input_train_poly = poly.fit_transform(input_train)
# input_test_poly = poly.transform(input_test)
#
# model_poly = LinearRegression()
# model_poly.fit(input_train_poly, output_train)
# pred_output_poly = model_poly.predict(input_test_poly)
#
# print(f"Auto calculated r2_score with poly: ", model_poly.score(input_test_poly,output_test))
# print(f"Manually calculated r2_score with poly : ", r2_score(output_test,pred_output_poly))
# print(f"Mean Sqrd Error with poly : ", root_mean_squared_error(output_test,pred_output_poly))
# print(f"model params coeff with poly: { model_poly.coef_}, intercept: {model_poly.intercept_}")
#
#
# vif_data = pd.DataFrame()
# vif_data['Features'] = inputx.columns
# vif_data['VIF'] = [variance_inflation_factor(inputx.values,i) for i in range(inputx.shape[1])]
# print(vif_data)

# corr_matrix = inputx.corr()
# sns.heatmap(corr_matrix,annot=True)
# sns.boxplot(data=inputx)
# plot.show()


# # removing outliers
# TV_Budget_Q1 = df['TV Budget ($)'].quantile(0.2)
# TV_Budget_Q3 = df['TV Budget ($)'].quantile(0.8)
# # TV_Budget_IQR = TV_Budget_Q3 - TV_Budget_Q1
# TV_Budget = df[(df['TV Budget ($)'] >= TV_Budget_Q1) & (df['TV Budget ($)'] <= TV_Budget_Q3) ]
#
# Radio_Budget_Q1 = df['Radio Budget ($)'].quantile(0.2)
# Radio_Budget_Q3 = df['Radio Budget ($)'].quantile(0.8)
# # Radio_Budget_IQR = Radio_Budget_Q3- Radio_Budget_Q1
# Radio_Budget = df[(df['Radio Budget ($)'] >= Radio_Budget_Q1) & (df['Radio Budget ($)'] <= Radio_Budget_Q3) ]
#
# NewsPaper_Budget_Q1 = df['Newspaper Budget ($)'].quantile(0.2)
# NewsPaper_Budget_Q3 = df['Newspaper Budget ($)'].quantile(0.8)
# # NewsPaper_Budget_IQR = NewsPaper_Budget_Q3- NewsPaper_Budget_Q1
# Newspaper_Budget = df[(df['Newspaper Budget ($)'] >= NewsPaper_Budget_Q1) & (df['Newspaper Budget ($)'] <= NewsPaper_Budget_Q3) ]
#
# print(TV_Budget)
# print(Radio_Budget)
# print(Newspaper_Budget)

# model = LinearRegression()
# rfe = RFE(model,n_features_to_select=2)
# input_new= rfe.fit_transform(inputx, inputy)
# print(input_new)
# print(inputx.columns[rfe.support_])
# print(inputx.columns[rfe.ranking_])


# #implementing Ridge Linear Regression
# #create a scaler object
# scaler = StandardScaler()
# #scale the inout train and test data sets
# input_scaled_train = scaler.fit_transform(input_train)
# input_scaled_test = scaler.transform(input_test)
#
# ridge = Ridge(alpha=0.5)
# ridge.fit(input_scaled_train,output_train)
#
# pred_output = ridge.predict(input_scaled_test)
#
# # print to console the score pf the model
# print(f"Auto calculated r2_score using ridge: ", ridge.score(input_test,output_test))
# print(f"Manually calculated r2_score using ridge : ", r2_score(output_test,pred_output))
# print(f"Mean Sqrd Error : ridge", root_mean_squared_error(output_test,pred_output))
# print(f"model params coeff using ridge: { ridge.coef_}, intercept: {ridge.intercept_}")