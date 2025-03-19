import numpy as np
import pandas as pd

from keras.api.models import model_from_json
from sklearn.preprocessing import LabelEncoder, StandardScaler

#---------------function to encode the strings value to numeric values --------------
def label_encode_column(column):
    """

    :param column:
    :return: numeric encoding of the original value
    """
    le = LabelEncoder()
    return le.fit_transform(column)

#----------------load the saved ANN model---------------------------------
with open('./my_first_ANN_model.json', 'r') as json_file:
    loaded_model_json = json_file.read()
model = model_from_json(loaded_model_json)

#----------------------------------------------------------------------------
#---------load the file for precitions------------------------------
df = pd.read_csv('./HR_Data.csv')
#-------------Encode the input values--------------
df['Attrition'] = LabelEncoder().fit_transform(df['Attrition'])

#-----------crate the independent and dependent feature set------------------------
inputx = df.iloc[:, 1:39]
columns_to_encode = inputx.columns.tolist()

#----------encode string to numeric values----------------
inputx = inputx[columns_to_encode].apply(label_encode_column)
#------------------------------------------------------------------------------------

#---------scale the independent features for optimized prediction-----------------------
scaler = StandardScaler()
samples = inputx.iloc[3:, 0:39].to_numpy()
samples_scaled = scaler.fit_transform(samples)

#--------------predictions on sampled and scaled inputs ----------------------
predictions = model.predict(samples_scaled)
predicted_classes = (predictions > 0.5).astype(int)
print(predicted_classes.flatten())