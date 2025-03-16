import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV

# import dotenv
import numpy as np
# import matplotlib
# import matplotlib.pyplot as plot
# import seaborn as sns

df = pd.read_csv('./HR_Data.csv')
# df['JobRole'] = LabelEncoder().fit_transform(df['JobRole'])
# print(df['JobRole'])
# print(df)
inputx = df.iloc[:,1:39]
inputy = df.iloc[:,0]
print(inputx)
print(inputy.head())

# input_train, input_test,output_train, output_test = train_test_split(inputx, inputy,test_size=1/3,random_state=42)
# #create a scaler object
# scaler = StandardScaler()
#
#
# #scale the inout train and test data sets
# input_scaled_train = scaler.fit_transform(input_train)
# input_scaled_test = scaler.transform(input_test)
# classifier = KNeighborsClassifier(
#     n_neighbors=5,
#     p=2
# )
#
# params = {
#     'n_neighbors':[1,2,3,4,5],
#     'weights':['uniform', 'distance'],
#     'algorithm':['auto','ball_tree','kd_tree','brute'],
#     'p':[2],
#     'metric':['minkowski','euclidean']
#
# }
# grid_search = GridSearchCV(classifier, param_grid=params, cv=3)
# grid_search.fit(input_scaled_train,output_train)
# # Print the best parameters and score
# print("Best parameters:", grid_search.best_params_)
# print("Best cross-validation score:", grid_search.best_score_)
#
# # Evaluate on the test set
# best_model = grid_search.best_estimator_
# test_score = best_model.score(input_scaled_test, output_test)
# print("Test set accuracy:", test_score)
# # grid.
# # classifer.fit(input_scaled_train,output_train)
# # pred_output= classifer.predict(input_scaled_test)
# #
# # print(confusion_matrix(output_test,pred_output))
# # print(accuracy_score(output_test,pred_output))
# # print(classifer.predict([[35,5,5000,5,10]]))
