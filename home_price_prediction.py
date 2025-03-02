import pandas as pd
import numpy as np 
import matplotlib
import matplotlib.pyplot as plot
import seaborn as sns 
import dotenv , os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error


dotenv.load_dotenv()
matplotlib.use('Agg')

df = pd.read_csv('./home.csv',names=['Area','BedRooms','Price'])
# print(df)

# sns.pairplot(
#     data=df
# )
# plot.savefig('plt.png')

inputx = df.iloc[:,0:2]
inputy = df.iloc[0:,2]
# print(inputy)

input_train,input_test,output_train,output_test = train_test_split(inputx,inputy,test_size=1/4,random_state=0)
# print(input_train)
# print(output_train)

model = LinearRegression()
# print(f"params of model are : ", model.get_params())
model.fit(input_train,output_train)
pred_output = model.predict(input_test)
print(f"Auto calculated re2_score : ", model.score(input_test,output_test))
print(f"Manually calculated r2_score : ", r2_score(output_test,pred_output))
print(f"Mean Sqrd Error : ", root_mean_squared_error(output_test,pred_output))