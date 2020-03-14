from tensorflow.keras.layers import Dense, LSTM, Conv1D
from tensorflow.keras import Sequential
from tensorflow.keras.backend import clear_session
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import tanh, elu
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.models import load_model
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.metrics import RootMeanSquaredError
import wavio
import numpy as np
import matplotlib.pyplot as plt

class LSTAMP:
    def __init__(self,load_model = None, history=200, hidden_units = 36, learning_rate = 0.01, test_size=0.02):
        self.history = history
        self.hidden_units = hidden_units
        self.batch_size = 4096
        self.learning_rate = learning_rate
        if load_model == None:
            self.model = self.create()
        else:
            self.load(load_model)
        self.test_size = test_size
        
    def NormalizedRootMeanSquaredError(self,y_true, y_pred):
        return K.sqrt(K.sum(K.square(y_pred - y_true))/K.sum(K.square(y_true)))
    
    def create(self):
        conv = [35,12]
        clear_session()
        model = Sequential()
        model.add(Conv1D(conv[0], conv[1], strides=4, activation=elu, padding='same', input_shape=(self.history,1)))
        model.add(Conv1D(conv[0], conv[1], strides=3, activation=elu, padding='same'))
        model.add(LSTM(self.hidden_units))
        model.add(Dense(1, activation=tanh))
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss=mean_squared_error, metrics=[RootMeanSquaredError(),self.NormalizedRootMeanSquaredError])
        print('Created model:')
        print(model.summary())
        return model
    
    def normalize_wav(self,data):
        data_max = max(data)
        data_min = min(data)
        data_norm = max(data_max,abs(data_min))
        return data / data_norm
     
    def load_data(self, input_file = 'input', target_file = 'target'):

        X = np.array(wavio.read(f'Samples/{input_file}.wav').data[:,0].flatten())
        self.X = self.normalize_wav(X).reshape(len(X),1)
        y = np.array(wavio.read(f'Samples/{target_file}.wav').data[:,0].flatten())[:len(X)]
        self.y = self.normalize_wav(y).reshape(len(y),1)
        
        indices = np.arange(self.history) + np.arange(len(self.X)-self.history+1)[:,np.newaxis]
        self.X_ordered = tf.gather(self.X,indices)
        self.y_ordered = self.y[self.history-1:]
        shuffled_indices = np.random.permutation(len(self.X_ordered))
        self.X_random = tf.gather(self.X_ordered,shuffled_indices)
        self.y_random = tf.gather(self.y_ordered, shuffled_indices)
    
        plt.plot(self.y, label='Target')
        plt.plot(self.X, label='Input')
        plt.legend(loc=1)
        plt.title('Training Data')
        plt.show()
    
    def fit(self,epochs):
        self.load_data()
        self.model.fit(self.X_random,self.y_random, epochs=epochs, batch_size=self.batch_size, validation_split=self.test_size)
        
    def test(self):
        try:
            self.X_ordered
            self.y_ordered
        except:
            self.load_data()
        prediction = self.model.predict(self.X_ordered, batch_size=self.batch_size)
        nrmse = np.sqrt(np.sum((prediction - self.y_ordered)**2)/np.sum(self.y_ordered**2))
        print('NRMSE: {0:.4f}'.format(nrmse))
        plt.plot(self.y_ordered,label='Target')
        plt.plot(self.X[self.history-1:], label='Input' )
        plt.plot(prediction, label='Prediction')
        #plt.plot(X[history-1:], label='Input')
        plt.xlim(400000,401000)
        plt.legend(loc=1)
        plt.show()
        wavio.write('Samples/pred.wav', prediction, 44100, sampwidth=3)
        plt.subplot(1,3,1)
        powerSpectrumInput, frequenciesFoundInput, time, imageAxis = plt.specgram(self.X.flatten(), Fs=44100)
        plt.title('Input Spectrum')
        plt.xlabel('Time [s]')
        plt.ylabel('Frequency [Hz]')
        plt.subplot(1,3,2)
        powerSpectrumTarget, frequenciesFoundTarget, time, imageAxis = plt.specgram(self.y[self.history:].flatten(), Fs=44100)
        plt.title('Target Spectrum')
        plt.xlabel('Time [s]')
        plt.subplot(1,3,3)
        powerSpectrumPrediction, frequenciesFoundPrediction, time, imageAxis = plt.specgram(prediction.flatten(), Fs=44100)
        plt.title('Prediction Spectrum')
        plt.xlabel('Time [s]')
        plt.tight_layout()
        plt.show()
    #    Z = 10. * np.log10(powerSpectrumTarget) - 10. * np.log10(powerSpectrumPrediction)
    #    Z = np.flipud(Z)
    #    difplot = plt.pcolormesh(time, frequenciesFoundPrediction, Z)
    #    plt.colorbar(difplot)
    #    plt.title('Spectrum Difference')
    #    plt.xlabel('Time [s]')
    #    plt.ylabel('Frequencies [Hz]')
    #    plt.tight_layout()
    #    plt.show()
    
    def save(self,filename):
        self.model.save(f'{filename}.h5')
        print(f'Saved LSTAMP to {filename}.h5')
    
    def load(self,filename):
        self.model = load_model(f'{filename}.h5', custom_objects={'NormalizedRootMeanSquaredError' : self.NormalizedRootMeanSquaredError})
        self.history = self.model.input.shape[1]
        print(f'Loaded model {filename}:')
        print(self.model.summary())
        
    def amp(self,filename):
        amp_X = wavio.read(f'Samples/{filename}.wav').data.flatten()
        amp_X = self.normalize_wav(amp_X).reshape(len(amp_X),1)
        
        indices = np.arange(self.history) + np.arange(len(amp_X)-self.history+1)[:,np.newaxis]
        X_ordered = tf.gather(amp_X,indices)
        
        amp_pred = self.model.predict(X_ordered, batch_size = self.batch_size)
        
        plt.plot(amp_pred, label='Output')
        plt.plot(amp_X, label='Input')
        plt.legend(loc=1)
        plt.show()
        wavio.write('Samples/amped.wav', amp_pred, 44100, sampwidth=3)
        