from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import Adamax

def build_model(maxlen, char_len):
    print('Build model...')
    model = Sequential()
    model.add(LSTM(512, input_shape=(maxlen, char_len), return_sequences=True, activation='sigmoid'))
    model.add(Dropout(0.2))
    model.add(LSTM(256, activation='sigmoid', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(128, activation='sigmoid'))
    model.add(Dropout(0.2))
    model.add(Dense(char_len))
    model.add(Activation('softmax'))
    optimizer = Adamax()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return model

def train(model, X, y, p_batch_size = 1024, p_epoch=1):
    model.fit(X, y, batch_size=p_batch_size, nb_epoch=p_epoch)

def save_model(model, filename):
    model.save_weights(filename)

def load_model(model, filename):
    model.load_weights(filename)




