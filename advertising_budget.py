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


# Enable reading of environment variable
dotenv.load_dotenv()

df = pd.read_csv('./Ad_budget.csv')
# print(df['Sales (units)'])
inputx = df.iloc[:,1:4]
inputy = df.iloc[:,4]
# print(df)
# print(inputx)
# print(inputy)

# Supervised learning
# split the data into train and test
input_train,input_test,output_train,output_test = train_test_split(inputx,inputy,test_size=1/4,random_state=0)
#

# create a model
model = LinearRegression()

# train the regression model
model.fit(input_train,output_train)
pred_output = model.predict(input_test)

# print to console the score pf the model
print(f"Auto calculated r2_score : ", model.score(input_test,output_test))
print(f"Manually calculated r2_score : ", r2_score(output_test,pred_output))
print(f"Mean Sqrd Error : ", root_mean_squared_error(output_test,pred_output))
print(f"model params coeff: { model.coef_}, intercept: {model.intercept_}")


#create a scaler object
scaler = StandardScaler()

#scale the inout train and test data sets
input_scaled_train = scaler.fit_transform(input_train)
input_scaled_test = scaler.transform(input_test)

# train the model with scaling
model.fit(input_scaled_train, output_train)
pred_output2 = model.predict(input_scaled_test)

print(f"Auto calculated r2_score with scaling : ", model.score(input_scaled_test,output_test))
print(f"Manually calculated r2_score with scaling : ", r2_score(output_test,pred_output2))
print(f"Mean Sqrd Error with scaling : ", root_mean_squared_error(output_test,pred_output2))
print(f"model params coeff with scaling: { model.coef_}, intercept: {model.intercept_}")
# #
poly = PolynomialFeatures(degree=2, include_bias=False)
input_train_poly = poly.fit_transform(input_train)
input_test_poly = poly.transform(input_test)

model_poly = LinearRegression()
model_poly.fit(input_train_poly, output_train)
pred_output_poly = model_poly.predict(input_test_poly)

print(f"Auto calculated r2_score with poly: ", model_poly.score(input_test_poly,output_test))
print(f"Manually calculated r2_score with poly : ", r2_score(output_test,pred_output_poly))
print(f"Mean Sqrd Error with poly : ", root_mean_squared_error(output_test,pred_output_poly))
print(f"model params coeff with poly: { model_poly.coef_}, intercept: {model_poly.intercept_}")
#
#
# corr_matrix = inputx.corr()
# sns.heatmap(corr_matrix,annot=True)
# sns.boxplot(data=inputx)
# plot.show()
#
