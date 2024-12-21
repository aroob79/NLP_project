import gensim
import gensim.downloader as api
from nltk import WordNetLemmatizer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report


lemmatizer = WordNetLemmatizer()
# load the file
path_txt_file = r'E:\python\basic_code\automate the boring stuf\NLP\SMS_spam_classifier\SMSSpamCollection.txt'
texts = pd.read_csv(path_txt_file,
                    sep='\t', names=['label', 'text'])

# preprocess the text
corpus = []
for line in texts['text']:
    temp = gensim.utils.simple_preprocess(line)
    temp = [lemmatizer.lemmatize(word) for word in temp]
    corpus.append(temp)

# init the model
models = gensim.models.Word2Vec(window=10, min_count=2, workers=4)
models.build_vocab(corpus, progress_per=1000)
print('....training the model...')
models.train(corpus, epochs=30, compute_loss=True,
             total_examples=models.corpus_count,)

# preprocess the text for classifier

avg_vect = []
nan_indx = []
for i, sent in enumerate(corpus):
    wave_sent = [models.wv[word]
                 for word in sent if word in models.wv.index_to_key]
    if len(wave_sent) == 0:
        nan_indx.append(i)
    else:
        avg_vect.append(np.mean(wave_sent, axis=0))
avg_vect = np.array(avg_vect)
labels = texts['label']
encoded_labels = pd.get_dummies(labels)['ham']
encoded_labels = encoded_labels[~encoded_labels.index.isin(nan_indx)]

xtr, xtest, ytr, ytest = train_test_split(
    avg_vect, encoded_labels, test_size=0.3, random_state=79)

# train the classifier
print('... train the classifier ....')
xgbcls = XGBClassifier()
xgbcls.fit(xtr, ytr)

pred_clss = xgbcls.predict(xtest)
print(classification_report(ytest, pred_clss))
