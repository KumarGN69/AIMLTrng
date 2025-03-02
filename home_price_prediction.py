import pandas as pd
import numpy as np 
import matplotlib
import matplotlib.pyplot as plot
import seaborn as sns 
import dotenv , os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures



dotenv.load_dotenv()
matplotlib.use('Agg')

df = pd.read_csv('./home.csv',names=['Area','BedRooms','Price'])
# print(df)

# sns.pairplot(
#     data=df
# )
# plot.savefig('plt.png')

sns.scatterplot(x=df['Area'],y=df['Price'])
plot.savefig(f'plt_area.png')

sns.scatterplot(x=df['BedRooms'],y=df['Price'])
plot.savefig(f'plt_br.png')

inputx = df.iloc[:,0:2]
inputy = df.iloc[0:,2]
# print(inputy)

#Supervised learning
# split the data into train and test
input_train,input_test,output_train,output_test = train_test_split(inputx,inputy,test_size=1/4,random_state=0)

#create a model
model = LinearRegression()

# train the regression model
model.fit(input_train,output_train)
pred_output = model.predict(input_test)

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





