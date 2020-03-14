from tensorflow.keras.layers import Dense, LSTM, GRU, Conv1D
from tensorflow.keras import Sequential
from tensorflow.keras.backend import clear_session
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import tanh, elu
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.models import load_model
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.metrics import RootMeanSquaredError
from scipy import signal
from scipy.io import wavfile

#from tensorflow.keras.models import load_model
import wavio
import numpy as np
import matplotlib.pyplot as plt

amp_name = 'Silverface'
history = 200
batch_size = 4096
hidden_units = 36
learning_rate = 0.01
test_size = 0.2

def NormalizedRootMeanSquaredError(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_pred - y_true))/K.sum(K.square(y_true)))
    

clear_session()
model = Sequential()
model.add(Conv1D(35, 12,strides=4, activation=elu, padding='same',input_shape=(history,1)))
model.add(Conv1D(35, 12,strides=3, activation=elu, padding='same'))
model.add(LSTM(hidden_units))
#model.add(LSTM(hidden_units,input_shape=(history,1)))
#model.add(Dropout(0.2))
model.add(Dense(1, activation=tanh))
model.compile(optimizer=Adam(learning_rate=learning_rate), loss=mean_squared_error, metrics=[RootMeanSquaredError(),NormalizedRootMeanSquaredError])
print(model.summary())

def normalize_wav(data):
    data_max = max(data)
    data_min = min(data)
    data_norm = max(data_max,abs(data_min))
    return data / data_norm
    
X = np.array(wavio.read('Samples/strat_input.wav').data[:,0].flatten())
X = normalize_wav(X).reshape(len(X),1)
y = np.array(wavio.read('Samples/strat_target.wav').data[:,0].flatten())[:len(X)]
y = normalize_wav(y).reshape(len(y),1)

indices = np.arange(history) + np.arange(len(X)-history+1)[:,np.newaxis]
X_ordered = tf.gather(X,indices)
y_ordered = y[history-1:]
shuffled_indices = np.random.permutation(len(X_ordered))
X_random = tf.gather(X_ordered,shuffled_indices)
y_random = tf.gather(y_ordered, shuffled_indices)

plt.plot(y, label='Target')
plt.plot(X, label='Input')
plt.legend(loc=1)
plt.show()

def fit(epochs):
    model.fit(X_random,y_random, epochs=epochs, batch_size=batch_size, validation_split=test_size)
    
def predict():
    prediction = model.predict(X_ordered, batch_size=batch_size)
    nrmse = np.sqrt(np.sum((prediction - y_ordered)**2)/np.sum(y_ordered**2))
    print('NRMSE: {0:.4f}'.format(nrmse))
    plt.plot(y_ordered,label='Target')
    plt.plot(X[history-1:], label='Input' )
    plt.plot(prediction, label='Prediction')
    #plt.plot(X[history-1:], label='Input')
    plt.xlim(400000,401000)
    plt.legend(loc=1)
    plt.show()
    wavio.write('Samples/prediction_lstm_sweep.wav', prediction, 44100, sampwidth=3)
    plt.subplot(1,3,1)
    powerSpectrumInput, frequenciesFoundInput, time, imageAxis = plt.specgram(X.flatten(), Fs=44100)
    plt.title('Input Spectrum')
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.subplot(1,3,2)
    powerSpectrumTarget, frequenciesFoundTarget, time, imageAxis = plt.specgram(y[history:].flatten(), Fs=44100)
    plt.title('Target Spectrum')
    plt.xlabel('Time [s]')
    plt.subplot(1,3,3)
    powerSpectrumPrediction, frequenciesFoundPrediction, time, imageAxis = plt.specgram(prediction.flatten(), Fs=44100)
    plt.title('Prediction Spectrum')
    plt.xlabel('Time [s]')
    plt.tight_layout()
    plt.show()
    Z = 10. * np.log10(powerSpectrumTarget) - 10. * np.log10(powerSpectrumPrediction)
    Z = np.flipud(Z)
    difplot = plt.pcolormesh(time, frequenciesFoundPrediction, Z)
    plt.colorbar(difplot)
    plt.title('Spectrum Difference')
    plt.xlabel('Time [s]')
    plt.ylabel('Frequencies [Hz]')
    plt.tight_layout()
    plt.show()

def save():
    model.save(f'{amp_name}-{history}-{hidden_units}.h5')

def load():
    global model
    model = load_model('Silverface-200-36.h5', custom_objects={'NormalizedRootMeanSquaredError' : NormalizedRootMeanSquaredError})

def amp():
    amp_X = wavio.read('Samples/test_input.wav').data.flatten()
    amp_X = (amp_X/max([max(amp_X),abs(min(amp_X))])).reshape(len(amp_X),1)
    
    reshaped_X = []
    reshaped_y = []
    for i in range(len(amp_X)-history):
        temp_X = amp_X[i:i+history]
        reshaped_X.append(np.array(temp_X))
    amp_X_shaped = np.array(reshaped_X)
    
    amp_pred = model.predict(amp_X_shaped, batch_size = batch_size)
    
    plt.plot(amp_pred, label='Prediction')
    plt.plot(amp_X, label='Input')
    plt.legend(loc=1)
    plt.show()
    wavio.write('amp_pred.wav', amp_pred, 44100, sampwidth=3)
    

