import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences

from keras.api.models import Model, Sequential
from keras.api.layers import Embedding, LSTM, Dense
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

#----------function to encode different string values to numeric valid------------
def label_encode_column(column):
    le = LabelEncoder()
    return le.fit_transform(column)

# ----------------to create an embedding index from the pretrained Glove embedding------------
def load_glove_embedding(glove_path):
    embedding_index = {}
    with open(glove_path, encoding='utf8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = vector
    return embedding_index

#--------------read the labelled CSV file -----------------------
df = pd.read_csv("./classified_neutral_posts.csv")

# ------------------separate the independent and dependent features----------------
inputx = df.iloc[:, 0:4]
inputy = df.iloc[0:, 4]
print(inputx, inputy)

#---------------------split the train and test data------------------------
input_train, input_test, output_train, output_test = train_test_split(inputx, inputy, test_size=0.2, stratify=inputy, random_state=42)

#------------------create the list of corpus on which the embeddings need to trained--------------------
input_train['summary'] = input_train['summary'].astype(str)
input_train_sentences = [input_train['summary'].iloc[record] for record in range(0,input_train['summary'].size)]


#---------------------------tokenize the training corpus---------------------
train_tokenizer = Tokenizer()
train_tokenizer.fit_on_texts(input_train_sentences)
word_index = train_tokenizer.word_index
vocab_size = len(word_index) + 1

#--------------------------create the token sequences for training----------------------
input_train_sequences = train_tokenizer.texts_to_sequences(input_train_sentences)
input_train_paddings = pad_sequences(input_train_sequences, padding='post')

#------------load the pre trained Glove embedding vectors--------------------
glove_path = "./glove.6B.50d.txt"
embedding_index = load_glove_embedding(glove_path)

#---------create the embedding matrix-----------------------
embedding_dim = 50 # dimension is 50 because the pretrained Glove embedding file has 50 dimension vector---------
embedding_matrix = np.zeros((vocab_size, embedding_dim))

#--------------------------create a matrix for pretrained embedding matrix-----------------
for word, idx in word_index.items():
    vector = embedding_index.get(word)
    if vector is not None:
        embedding_matrix[idx] = vector

#-----------------------------create an ANN model and add layers-------------------
model = Sequential()
model.add(Embedding(
            input_dim=vocab_size, # input dimension is the number of words in the pretrained embedding file
            output_dim=embedding_dim, # Output dimension is the dimension of the vector matrix in the pretrained embedding file
            weights=[embedding_matrix], # weights to used for tuning the NN Model using the vectors from pretrained embeddings
            input_length=input_train_paddings.shape[1],
            trainable=False
        )
    )
model.add(LSTM(64))
model.add(Dense(len(set(output_train)),activation='softmax')) # output layer is set to number of class labels in the training set
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()

#---------------------create a list of train labels and using Label binarizer to convert training text to numerical values--------------------------------
train_labels_list =[output_train.iloc[record] for record in range(0,output_train.size)]
label_binarizer = LabelBinarizer()
binary_train_labels = label_binarizer.fit_transform(train_labels_list) # use fit transform on the training set only

#-------------------training the learning algorithm using the pretrained embeddings and the sample data set--------------
model.fit(input_train_paddings,binary_train_labels, epochs=10, verbose=1)

#--------------------------create the token sequences for testing----------------------
input_test['summary'] = input_test['summary'].astype(str)
input_test_sentences = [input_test['summary'].iloc[record] for record in range(0,input_test['summary'].size)]

test_tokenizer = Tokenizer()
test_tokenizer.fit_on_texts(input_test_sentences)
input_test_sequences = test_tokenizer.texts_to_sequences(input_test_sentences)
input_test_paddings = pad_sequences(input_test_sequences, padding='post')

#---------------------create a list of test labels and using Label binarizer to convert test text to numerical values-------------
output_test_labels = [output_test.iloc[record] for record in range(0,output_test.size)]
binary_test_labels = label_binarizer.transform(output_test_labels) # use the same label binarizer as used for training labels

#------------evaluate the model accuracy--------------------------------
loss, accuracy = model.evaluate(input_test_paddings, binary_test_labels)
print(f"Accuracy : {accuracy * 100:2f}")

# --------------measure the prediction accuracy---------------------
predictions = model.predict(input_test_paddings)
predicted_indices = np.argmax(predictions, axis=1)
predicted_labels = label_binarizer.classes_[predicted_indices]
print(classification_report(output_test_labels, predicted_labels))
print(confusion_matrix(output_test_labels, predicted_labels))
#--------------save the model to a json file-------------
model_json = model.to_json()
with open("selflearning_pretrained_Glove_text_embeddings_model.json", "w") as json_file:
    json_file.write(model_json)

# -------------------Plot with Seaborn and save it to file -----------------
matplotlib.use('Agg') # for saving it to an image file
plt.figure(figsize=(8, 6), dpi=100)
ax = plt.axes()
ax.set_xlabel("Predicted Labels", fontsize=12)
ax.set_ylabel("Actual Labels", fontsize=12)
ax.set_title("Confusion Matrix", fontsize=14, pad=20)
sns.heatmap(confusion_matrix(output_test_labels, predicted_labels), annot=True)
# plt.show()
# Customize labels


# # Save as image file
plt.savefig('confusion_matrix.png', bbox_inches='tight', dpi=300)
plt.close()  # Close the plot to free memory

