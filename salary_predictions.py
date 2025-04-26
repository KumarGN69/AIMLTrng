import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plot

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error

# read and load the data from file
salary_data = pd.read_csv('./Salary_Data.csv')
salary_data_filtered = salary_data.iloc[:,4:6]

# remove rows with no or zero values
salary_data_filtered = salary_data_filtered[(salary_data_filtered != 0).all(axis=1)]
salary_data_filtered = salary_data_filtered.dropna()

#filter the data frame to retain data of only dependent and independentfeatures
inputx = salary_data_filtered.iloc[:,0:1]
inputy = salary_data_filtered.iloc[:,1:2]

# Supervised learning
input_train, input_test, output_train, output_test = train_test_split(inputx, inputy, test_size = 1/5, random_state=42)
model = LinearRegression()
model.fit(input_train, output_train)
pred_output= model.predict(input_test)

print(f"Auto calculated r2_score : ", model.score(input_test,output_test))
print(f"Manually calculated r2_score : ", r2_score(output_test,pred_output))
print(f"Mean Sqrd Error : ", root_mean_squared_error(output_test,pred_output))
print(f"model params coeff: { model.coef_}, intercept: {model.intercept_}")

# visualize the model
plot.scatter(input_train,output_train, color='blue', label='Training Data')
plot.plot(input_test,pred_output, color='red', label='Predicted Data')
plot.xlabel('Years of Experience')
plot.ylabel('Salary')
plot.savefig('salary_pred_output.png')
plot.legend()
plot.show()
plot.close()

#predict for a given input
years = float(input("Enter Years of Experience: "))
if years < 0:
    print("Please enter a positive number.")
else:
    sample_data = {
    "Years of Experience": [years]
}
model.predict(pd.DataFrame(sample_data))
print(f"Approximate salary for {years} years of experience is: {int(model.predict(pd.DataFrame(sample_data))[0][0])}")