# Tube-Amp-Modeller
A neural network designed to model a guitar tube amplifier inspired by a project by @sdatkinson.

Since Tube Amplifiers typically act non-linearly to the input signal, it becomes an interesting problem to model this using a neural network. LSTM networks are notoriously suited for this kind of task.

The output signal of the amplifier depends on a 'history' h of input samples:

y<sub>t</sub> = a<sub>1</sub> x<sub>t</sub> + a<sub>2</sub> x<sub>t-1</sub> + ... + a<sub>h</sub> x<sub>t-h</sub>

Hence we need to create a data set that inputs h samples for every 1 output sample.

The code is written to be executed in an IPython console.

## Training

Various parameters can be set in the code before execution. The standard parameters for the included samples are:
```
history = 200
batch_size = 4096
hidden_units = 36
conv = [35,12]
learning_rate = 0.01
test_size = 0.2
```
Once the code is executed LSTAMP can be instantiated using:
```
LSTAMP = LSTAMP()
```
when including parameter load_model, one can load a pretrained model.

Once an LSTAMP has been made it can be trained for n epochs using
```
LSTAMP.fit(n)
```
Once trained, the model can be saved using
```
LSTAMP.save(filename)
```

## Amplifying using a trained model

Once a model has been trained and saved it can be used to amplify audio samples.
```
LSTAMP.amp(filename)
```
to amplify filename.wav.

## Results
![ResampleTest](plot.png)

As can be seen in the above plot, the model is able to come close to the target after 200 epochs. Although there's visibly more detail in the target file (more harmonics), this difference seems to be inaudible ben playing back these results. One interesting find is that when skipping every other input signal (reducing the input size) to improve speed, the results are still very good.
