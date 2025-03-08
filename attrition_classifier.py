import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score

# import dotenv
import numpy as np
# import matplotlib
# import matplotlib.pyplot as plot
# import seaborn as sns

df = pd.read_csv('./employee_attrition.csv')
df['JobRole'] = LabelEncoder().fit_transform(df['JobRole'])
# print(df['JobRole'])
print(df)
inputx = df.iloc[:,0:5]
inputy = df.iloc[:,5]
print(inputx)
print(inputy)

input_train, input_test,output_train, output_test = train_test_split(inputx, inputy,test_size=1/3,random_state=42)
#create a scaler object
scaler = StandardScaler()


#scale the inout train and test data sets
input_scaled_train = scaler.fit_transform(input_train)
input_scaled_test = scaler.transform(input_test)
classifer = KNeighborsClassifier(
    n_neighbors=5,
    p=2
)
classifer.fit(input_scaled_train,output_train)
pred_output= classifer.predict(input_scaled_test)

print(confusion_matrix(output_test,pred_output))
print(accuracy_score(output_test,pred_output))
print(classifer.predict([[35,5,5000,5,10]]))
