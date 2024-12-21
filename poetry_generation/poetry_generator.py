
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot as plt


poetry = ["""She walks in beauty, like the night 
Of cloudless climes and starry skies; 
And all that’s best of dark and bright
Meet in her aspect and her eyes; 
Thus mellowed to that tender light
Which heaven to gaudy day denies,
Love is a fire that burns unseen, 
a wound that aches yet isn’t felt, 
an always discontent contentment, 
a pain that rages without hurting,
Upon my word, I tell you faithfully
Through life and after death you are my queen;
For with my death the whole truth shall be seen.
Your two great eyes will slay me suddenly;
Their beauty shakes me who was once serene;
Straight through my heart the wound is quick and keen.
love will hurt you but 
love will never mean to 
love will play no games
cause love knows life 
has been hard enough already.
And I will not cry also 
Although you will expect me to
I was wiser too than you had expected 
For I knew all along you were mine.
We might be fifty, we might be five,
So snug, so compact, so wise are we!
Under the kitchen-table leg
My knee is pressing against his knee.
Our shutters are shut, the fire is low,
The tap is dripping peacefully;
The saucepan shadows on the wall
Are black and round and plain to see.
"""]

texts = [txt.lower().split('\n') for txt in poetry]
tokenizer = Tokenizer(oov_token='<OOV>')
tokenizer.fit_on_texts(texts[0])

vocab = tokenizer.word_index
vocab_size = len(vocab)+1

# convert the test into sequence
input_seq = []
for line in texts[0]:
    seq = tokenizer.texts_to_sequences([line])[0]
    for indx in range(1, len(seq)):
        input_seq.append(seq[:indx+1])
sequences = pad_sequences(input_seq, padding='pre')
max_len = np.shape(sequences)[1]

info = {'tokenizer': tokenizer, 'vocab_size': vocab_size, 'max_len': max_len}
# saving the tokenizer
print("saving the tokenizer....")
with open('info.pkl', 'wb') as file:
    pickle.dump(info, file)
file.close()

# slit into x and y
x_d = sequences[:, :-1]
y_d = sequences[:, -1]

# convert the label into categorical encoding
lebels = to_categorical(y_d, num_classes=vocab_size)

# define the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=500),
    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(200, return_sequences=True)),
    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(100, return_sequences=True)),
    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(80, return_sequences=True)),
    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(60)),
    tf.keras.layers.Dense(50),
    tf.keras.layers.Dense(40),
    tf.keras.layers.Dense(30),
    tf.keras.layers.Dense(vocab_size, activation='softmax')
])


# training the model
model.compile(loss=CategoricalCrossentropy(),
              optimizer=Adam(0.01), metrics=['accuracy'])
summary = model.fit(x_d, lebels, epochs=200, verbose=1)

# saving the model
print("saving the model .....")
model.save('poetry_model.h5')
# plt the traning accuracy
accuracy = summary.history['accuracy']
plt.plot(range(len(accuracy)), accuracy)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.tight_layout()
plt.show()
