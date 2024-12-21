import pandas as pd
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, classification_report
import pickle

physical_devices = tf.config.list_physical_devices("GPU")

# Check if there's a GPU available
if len(physical_devices) > 0:
    # Set memory growth for the GPU
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("GPU memory growth set successfully")
    except:
        print("Error setting GPU memory growth")
else:
    print("No GPU available. Running on CPU.")

# load the dataset

train_dt = pd.read_csv('train.csv', encoding='latin1',
                       skipinitialspace=True, skip_blank_lines=True)
train_dt = train_dt.dropna(how='all')
test_dt = pd.read_csv('test.csv', encoding='latin1',
                      skipinitialspace=True, skip_blank_lines=True)
test_dt = test_dt.dropna(how='all')

# getting the traning and test data
train_texts = train_dt['text'].astype(str).to_list()
train_sentimates = train_dt['sentiment']
# test data
test_texts = test_dt['text'].astype(str).to_list()
test_sentimates = test_dt['sentiment']

# initiate the token
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(train_texts)
vocab_size = len(tokenizer.word_index)+1
# converrt train text to sentence
train_sequence = tokenizer.texts_to_sequences(train_texts)
max_len = max([len(seq) for seq in train_sequence])
# padding
train_sequence = pad_sequences(train_sequence, padding='post',
                               maxlen=max_len, truncating='post')
# Save tokenizer to a file
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

# test
test_sequence = tokenizer.texts_to_sequences(test_texts)
# padding
test_sequence = pad_sequences(test_sequence, padding='post',
                              maxlen=max_len, truncating='post')

# processing lebels
class_names = np.unique(train_sentimates.to_numpy())
num_class = len(class_names)
class_num_dict = dict(zip(class_names, np.arange(num_class)))
num_class_dict = dict(zip(np.arange(num_class), class_names))
with open('numclass.pkl', 'wb') as f:
    pickle.dump(num_class_dict, f)


train_labels = train_sentimates.apply(lambda x: class_num_dict[x]).to_numpy()
test_labels = test_sentimates.apply(lambda x: class_num_dict[x]).to_numpy()

# initiate the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=200),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(30),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(loss=SparseCategoricalCrossentropy(), optimizer=Adam())
model.fit(train_sequence, train_labels, epochs=40)

##
print("saving the weights....")
model.save('saved_model.h5')

prediction = model.predict(test_sequence)
pred_label = tf.argmax(prediction, axis=1)
score = accuracy_score(test_labels, pred_label)

print(f'Accuracy of the model :{score}')

report = classification_report(test_labels, pred_label)
print(f' classification report : \n {report}')
