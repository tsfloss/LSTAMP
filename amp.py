import wavio
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import load_model

model_name = 'reaper'
model_file = 'hist-512-layers-[32, 16]-bs-1024-lr-0.001'
history = 512
test = np.load('train.npy')
X_test = test[0]
Y_test = np.insert(test[1],0,0)[:-1]

model = load_model('{}\\Models\\{}.h5'.format(model_name,model_file))

train_data = TimeseriesGenerator(X_test,Y_test,length=history,batch_size=1024)

prediction = model.predict(train_data)

rate = 44100
wavio.write("{}\\pred-{}.wav".format(model_name,model_file),prediction,rate,sampwidth=3)

plt.plot(Y_test[history:],label='target')
plt.plot(X_test[history-1:],label='input')
plt.plot(prediction,label='prediction')
plt.legend()
plt.show()
