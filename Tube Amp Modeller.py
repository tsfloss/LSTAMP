from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from tensorflow.keras.backend import clear_session
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import relu, tanh
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.models import load_model
import wavio
import matplotlib.pyplot as plt
import numpy as np
import os

class modeller:
    def __init__(self, amp_name='emissary', input_file = 'Samples/input.wav', target_file = 'Samples/target.wav', history=4096, dense_layers=[1024,512,256,256], lr=0.001, batch_size=4096, test_size=0.2):
        self.amp_name = amp_name
        self.history = history
        self.batch_size = batch_size
        self.dense_layers = dense_layers
        self.lr = lr
        self.model = self.initialize_model()
        self.input_file = input_file
        self.target_file = target_file
        self.test_size = test_size
    
    def create_data(self):
        x = np.array(wavio.read(self.input_file).data.flatten())
        x = x/max(x)
        y = np.array(wavio.read(self.target_file).data.flatten())[:len(x)]
        y = y/max(y)
        plt.plot(x)
        plt.plot(y)
        plt.show()
        train = np.stack((x,y))
        return train
    
    def load_data(self):
        train = self.create_data()
        X = train[0]
        y = np.insert(train[1],0,0)[:-1]
        train_frac = round(1 - self.test_size * len(X))
    
        X_train =X[:train_frac]
        X_test = X[train_frac:]
        y_train = y[:train_frac]
        y_test = y[train_frac:]
    
        self.train_data = TimeseriesGenerator(X_train, y_train, length = self.history, batch_size = self.batch_size)
        self.test_data = TimeseriesGenerator(X_test, y_test, length = self.history, batch_size = self.batch_size)
    
    def initialize_model(self):
        clear_session()
        model = Sequential()
        model.add(Dense(self.dense_layers[0], activation=relu, input_dim=self.history))
#        for i in range(1,len(self.dense_layers)):
#            model.add(Dense(self.dense_layers[i], activation=relu))
        [model.add(Dense(self.dense_layers[i], activation=relu)) for i in range(1,len(self.dense_layers))]
        model.add(Dense(1 ,activation=tanh))
        model.compile(optimizer=Adam(learning_rate=self.lr), loss=mean_squared_error, metrics=[RootMeanSquaredError()])
        print(model.summary())
        return model
#    
#    log_dir = "{}\\logs\\hist-{}-layers-{}-bs-{}-lr-{}".format(model_name,history,dense_layers,batch_size,learning_rate)
#    
#    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
#    callbacks=[tensorboard_callback]
#    
        
    def fit_model(self):
        self.load_data()   
        self.model.fit_generator(self.train_data ,epochs=200, shuffle=True, validation_data=self.test_data, validation_freq = 100)

    def save_model(self):
        if ~(os.path.exists('{}\\Models\\'.format(self.amp_name))):
            os.makedirs('{}\\Models'.format(self.amp_name))
        self.model.save("{}\\Models\\hist-{}-layers-{}-bs-{}-lr-{}.h5".format(self.amp_name,self.history,self.dense_layers,self.batch_size,self.lr))
        
model = modeller('test')

def reamp(input_file, amp_name, amp_file=None):
    if amp_file != None:
        model.model = load_model('{}\\Models\\{}.h5'.format(amp_name, amp_file))
    history = model.model.input_shape[1]
    X = np.array(wavio.read(input_file).data.flatten())
    X = X/(max(X))
    
    train_data = TimeseriesGenerator(X, X, length=history, batch_size=4096)
    prediction = model.model.predict(train_data)
    
    rate = 44100
    wavio.write("{}\\pred-{}.wav".format(amp_name, amp_file), prediction, rate, sampwidth=3)
    
    plt.plot(X[history-1:],label='Input')
    plt.plot(prediction,label='Prediction')
    plt.legend()
    plt.show()