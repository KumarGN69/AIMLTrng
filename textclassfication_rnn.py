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


df = pd.read_csv('./HR_Data.csv')
#-------------Encode the dependent feature--------------
df['Attrition'] = LabelEncoder().fit_transform(df['Attrition'])

#-----------crate the independent and dependent feature set------------------------
inputx = df.iloc[:, 1:39]
inputy = df.iloc[:, 0]
columns_to_encode = inputx.columns.tolist()

inputx = inputx[columns_to_encode].apply(label_encode_column)

input_train, input_test, output_train, output_test = train_test_split(inputx, inputy, test_size=1 / 3, random_state=42)
#create a scaler object
scaler = StandardScaler()
#scale the inout train and test data sets
input_scaled_train = scaler.fit_transform(input_train)
input_scaled_test = scaler.transform(input_test)

# --------instantiate a ANN model--------------
model = Sequential()
model.add(Dense(76, input_dim=38, activation='relu'))
model.add(Dense(57, activation='relu'))
model.add(Dense(38, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#--------------train the ANN model-------------
model.fit(input_scaled_train, output_train, epochs=100, batch_size=10, validation_data=(input_scaled_test, output_test))

loss, accuracy = model.evaluate(input_scaled_test, output_test)
print(f"Accuracy : {accuracy * 100:2f}")

#------------predict using the model-----------
samples = inputx.iloc[2:, 0:39].to_numpy()
samples_scaled = scaler.transform(samples)

predictions = model.predict(samples_scaled)

predicted_classes = (predictions > 0.5).astype(int)
print(predicted_classes.flatten())

#---------------------save the model in json format-------------------------------------
model_json = model.to_json()
with open("my_first_ANN_model.json", "w") as json_file:
    json_file.write(model_json)
