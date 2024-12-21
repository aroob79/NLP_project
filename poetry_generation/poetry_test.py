import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# load the tokenizer and import tant parameter
with open('info.pkl', 'rb') as file:
    info = pickle.load(file)
file.close()
tokenizer = info['tokenizer']
vacab_size = info['vocab_size']
max_len = info['max_len']
# 3loading the  model
model = tf.keras.models.load_model('poetry_model.h5')
words = tokenizer.index_word

inp_seq = ['love will hurt']
for _ in range(20):
    seq = tokenizer.texts_to_sequences(inp_seq)
    pad_seq = pad_sequences(seq, maxlen=max_len-1, padding='pre')
    pred_indx = tf.argmax(model.predict(pad_seq), axis=1)
    word = words.get(pred_indx[0].numpy(), 0)
    if word == 0:
        break
    inp_seq[0] += " "+word


print(inp_seq)
