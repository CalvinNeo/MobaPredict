from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense
from keras.optimizers import RMSprop
import numpy as np

max_features = 10000
max_len = 500

# save np.load
np_load_old = np.load
# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
# call load_data with allow_pickle implicitly set to true
(input_train, y_train), (test_data, test_labels) = imdb.load_data(num_words=10000)
# restore np.load for future normal usage
np.load = np_load_old

print "======="
print input_train.shape
print y_train.shape
print max_features
print len(input_train), 'train sequences' 
print input_train[0] 

model = Sequential()
model.add(Embedding(max_features, 128, input_length = max_len))
model.add(Conv1D(32, 7, activation = 'relu'))
model.add(MaxPooling1D(5))
model.add(Conv1D(32, 7, activation = 'relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(1))

model.summary()

model.compile(optimizer = RMSprop(lr = 1e-4),
             loss = 'binary_crossentropy',
             metrics = ['acc'])

