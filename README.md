# NextCharacterPrediction
I use Reuters data to train the model. I use an LSTM model. Input text length is 29 characters, A mapping is created by a dictionary over distinct characters used in the corpus. Depending on the device memory capacity, you have to use number of sentences from the corpus for training.

```python
# Get the reuters data
from nltk.corpus import reuters
```

```python
# overall algo:
# create a free flowing text without any punctuation etc. only alphabet
# reuters data consists of sentences; each sentence is a list of words
# so first of all combine all the words into one text string; then apply cleaning (only alphabet, 
# lower case, ' u  s ' to 'usa', etc)

# ok step 1
# combine all words of all sentences 
sentLimit = 1000 # adjust it depending on your device
sentCnt = 0
allText = ''
for sentence in reuters.sents():
    sentCnt += 1
    if sentCnt <= sentLimit:
        for word in sentence:
            allText += word + ' '
    else:
        break
# replace " ' s '" by " "
import re
allText = re.sub(" ' s ", " ", allText)
allText = re.sub(r'([^a-zA-Z])', " ", allText)
allText = allText.lower()
allText = re.sub(" u   s ", " usa ", allText)
allText = re.sub("  ", " ", allText)
allText = " ".join(allText.split())
```

```python
# now create sequences of seqLen (=5 say) characters where first four will be used for input and 5th char for output
allSeq = []
seqLen = 30
for i in range(0, len(allText)-seqLen+1):
    allSeq.append(allText[i:i+seqLen])
```

```python
# create a mapping of each character to an integer
map1 = sorted(list(set(allText)))
mapping = dict((c, i) for i, c in enumerate(map1))

# now convert allSeq using this mapping
allIntSeq = []
for i in range(len(allSeq)):
    allIntSeq.append([mapping[ch1] for ch1 in allSeq[i]])

# using numpy create X and y from allIntSeq
import numpy as np
X, y = np.array(allIntSeq)[:, :-1], np.array(allIntSeq)[:, -1]

# one-hot encode y to vector of lengthlen(map1)
from keras.utils import to_categorical
y = to_categorical(y, len(map1))

# split X and y into train and test
from sklearn.model_selection import train_test_split
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
```

```python
# now build the LSTM model: first import the libraries
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding

model = Sequential()
model.add(Embedding(len(map1), 50, input_length=29, trainable=True))
model.add(LSTM(150, recurrent_dropout=0.1, dropout=0.1))
model.add(Dense(len(map1), 'softmax'))
#print(model.summary)
model.compile(loss='categorical_crossentropy', metrics='acc', optimizer='adam')
model.fit(X_tr, y_tr, epochs=10, verbose=1, validation_data=(X_val, y_val))
```

```python
from tensorflow.keras.preprocessing.sequence import pad_sequences
# make prediction using seed_text
seed_text="but som"
numPred = 10

# make numPred predictions
for i in range(numPred):
    # convert to integers using mapping
    seedSeq = [mapping[ch1] for ch1 in seed_text]
    paddedSeq = pad_sequences([seedSeq], seqLen-1, truncating='pre')
    yhat_probs = model.predict(paddedSeq)
    for ch1, ind in mapping.items():
        if ind == np.argmax(yhat_probs):
            yhat = ch1
    seed_text += yhat
print(seed_text)
```
