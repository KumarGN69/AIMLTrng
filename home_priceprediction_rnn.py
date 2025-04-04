import keras.api.layers
import numpy as np
import pandas as pd
import csv
import tensorflow as tf
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences

from keras.api.models import Model, Sequential
from keras.api.layers import Input, LSTM, Dense

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


#
#----------function to encode different string values to numeric valid
def label_encode_column(column):
    le = LabelEncoder()
    return le.fit_transform(column)


df = pd.read_csv('./home.csv', names=['Area','BedRooms','Price'])
#-------------Encode the dependent feature--------------
# df['Attrition'] = LabelEncoder().fit_transform(df['Attrition'])
# print(df.columns)
# print(df)
# #-----------crate the independent and dependent feature set------------------------
inputx = df.iloc[:, 0:2]
inputy = df.iloc[:, 2]
# print(inputx, inputy)

input_train, input_test, output_train, output_test = train_test_split(inputx, inputy, test_size=1 / 3, random_state=42)
print(type(output_train.to_numpy()))
#create a scaler object
scaler_X = StandardScaler()
#scale the input train and test data sets
input_scaled_train = scaler_X.fit_transform(input_train)
input_scaled_test = scaler_X.transform(input_test)
# #
#create a scaler object
scaler_Y = StandardScaler()
#scale the input train and test data sets
output_scaled_train = scaler_Y.fit_transform(output_train.to_numpy().reshape(-1,1))
output_scaled_test = scaler_Y.transform(output_test.to_numpy().reshape(-1,1))
# #
# # --------instantiate a ANN model--------------
model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(1, activation='linear'))
#
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
#
# #--------------train the ANN model-------------
model.fit(input_scaled_train, output_scaled_train, epochs=1000, batch_size=2, validation_data=(input_scaled_test, output_scaled_test))
#
# print(output_test)
loss, mse = model.evaluate(input_scaled_test, output_scaled_test)
# print(model.metrics_names)
print(model.metrics_names[1])
print(f"loss : {loss}, mse: {mse}")
#
# #------------predict using the model-----------
samples = inputx.iloc[42:47, 0:2].to_numpy()
print(samples)
samples_scaled = scaler_X.transform(samples)
#
pred_scaled_output = model.predict(samples_scaled)
predictions = scaler_Y.inverse_transform(pred_scaled_output)
#
# predicted_classes = (predictions > 0.5).astype(int)
print(f"predictions :{predictions}")
# print(f"inverse transformed prediction {scaler.inverse_transform(predictions)}")
# #
# #---------------------save the model in json format-------------------------------------
model_json = model.to_json()
with open("my_home_price_prediction_ANN_model.json", "w") as json_file:
    json_file.write(model_json)
