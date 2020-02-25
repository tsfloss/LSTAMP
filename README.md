# Tube-Amp-Modeller
A neural network designed to model a guitar tube amplifier inspired by a project by @sdatkinson.

Since Tube Amplifiers typically act non-linearly to the input signal, it becomes an interesting problem to model this using a neural network.

The output signal of the amplifier depends on a 'history' h of input samples:

y<sub>t</sub> = a<sub>1</sub> x<sub>t</sub> + a<sub>2</sub> x<sub>t-1</sub> + ... + a<sub>h</sub> x<sub>t-h</sub>

Hence we need to create a data set that inputs h samples for every 1 output sample.
To do this we use the keras library TimeSeriesGenerator
