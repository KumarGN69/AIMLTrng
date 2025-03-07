import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

# import dotenv
import numpy as np
# import matplotlib
# import matplotlib.pyplot as plot
# import seaborn as sns

# enable loading of environment variable
# dotenv.load_dotenv()

df = pd.read_csv("./Social_Network_Ads.csv")
# print(df)
inputx= df.iloc[:,0:2]
inputy=df.iloc[:,2]
# print(inputx)

# supervised learning
# split the data into train and test
input_train,input_test,output_train,output_test = train_test_split(inputx,inputy,test_size=1/3,random_state=42)


#create a scaler object
scaler = StandardScaler()

#scale the inout train and test data sets
input_scaled_train = scaler.fit_transform(input_train)
input_scaled_test = scaler.transform(input_test)
# print(input_scaled_train)
# print(input_scaled_train, input_scaled_test)
# create a classifier model
classifier = KNeighborsClassifier(
    n_neighbors=5, # default
    metric="minkowski",
    p=2# for Euclidean distance
)
# train the classifier model
classifier.fit(input_scaled_train,output_train)

#predict the output
pred_output= classifier.predict(input_scaled_test)

print(confusion_matrix(output_test,pred_output))
print(accuracy_score(output_test,pred_output))

new_scaled_input = scaler.transform(np.array([[50,100000]]))
# print(new_scaled_input)
print(classifier.predict(new_scaled_input))


