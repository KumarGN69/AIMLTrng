from keras.api.models import model_from_json
import pandas as pd
import numpy as np
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import StandardScaler, LabelEncoder, LabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix

classification_labels = [
            'Ease of functionality',
            'User Experience',
            'App difficult to use',
            'Notification issues',
            'Software updates',
            'Audio quality issues',
            'Video quality Issues',
            'Connection Issues',
            'Pairing Issues',
            'Sync issues',
            'Voice Commands',
            'Android Auto Issues',
            'Bluetooth Issues',
            'Wifi Issues',
            'Price related',
        ]
#---------------binarizing the labels into which reviews need to be classified------------
label_binarizer = LabelBinarizer()
binary_train_labels = label_binarizer.fit_transform(classification_labels)
#----------------Load model arch and weights ------------------
with open("selflearning_model_arch.json") as json_file:
    loaded_model_json = json_file.read()

model = model_from_json(loaded_model_json)
model.load_weights("selflearning_model.weights.h5")


#------------Read reviews for file which need to be classified-----------

df = pd.read_csv('./classified_neutral_posts.csv')
inputx = df.iloc[:, 0:4]
review_sentences = [inputx['summary'].iloc[record] for record in range(0,inputx['summary'].size)]
inputy = df.iloc[0:, 4]

#-----------Tokenize the reviews that need to be classified-----------------------
test_tokenizer = Tokenizer()
test_tokenizer.fit_on_texts(review_sentences)
input_test_sequences = test_tokenizer.texts_to_sequences(review_sentences)
review_paddings = pad_sequences(input_test_sequences, padding='post')

predictions = model.predict(review_paddings)
predicted_indices = np.argmax(predictions, axis=1)
predicted_labels = label_binarizer.classes_[predicted_indices]

print(predicted_labels)

# print(classification_report(classification_labels, predicted_labels))
# print(confusion_matrix(classification_labels, predicted_labels))
