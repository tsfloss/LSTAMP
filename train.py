import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
import numpy as np
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import os

model_name = "reaper"
history = 1
batch_size=4096
dense_layers = [32,32]
learning_rate = 0.001

train = np.load('Samples\\{}.npy'.format(model_name))
X = train[0]
y = np.insert(train[1],0,0)[:-1]

train_frac = round(0.8 * len(X))

X_train =X[:train_frac]
X_test = X[train_frac:]
y_train = y[:train_frac]
y_test = y[train_frac:]

train_data = TimeseriesGenerator(X_train,y_train,length=history,batch_size=batch_size)
test_data = TimeseriesGenerator(X_test,y_test,length=history,batch_size =batch_size)

model = Sequential()
model.add(Dense(dense_layers[0], activation='relu', input_dim=history))
for i in range(1,len(dense_layers)):
    model.add(Dense(dense_layers[i], activation='relu'))
model.add(Dense(1 ,activation='tanh'))

optimiz = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimiz, loss= 'mean_squared_error', metrics=[tf.keras.metrics.RootMeanSquaredError()])

log_dir = "{}\\logs\\hist-{}-layers-{}-bs-{}-lr-{}".format(model_name,history,dense_layers,batch_size,learning_rate)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

model.fit_generator(train_data, epochs=1,shuffle=True,validation_data=test_data, callbacks=[tensorboard_callback])

if ~(os.path.exists('{}\\models\\'.format(model_name))):
    os.makedirs('{}\\models'.format(model_name))
model.save("{}\\models\\hist-{}-layers-{}-bs-{}-lr-{}.h5".format(model_name,history,dense_layers,batch_size,learning_rate))