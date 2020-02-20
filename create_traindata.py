import numpy as np
import wavio
import matplotlib.pyplot as plt

model_name = "reaper"
x = np.array(wavio.read('Samples/input.wav').data.flatten() / 2**23)
y = np.array(wavio.read('Samples/fuzz.wav').data.flatten() / 2**23)[::2][:len(x)]


print(x.shape)
print(y.shape)
plt.plot(x)
plt.plot(y)
plt.show()

train = np.stack((x,y))
print(train[0],train[1])
np.save('Samples\\{}.npy'.format(model_name), train)
