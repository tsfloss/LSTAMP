# Tube-Amp-Modeller
A neural network designed to model a guitar tube amplifier inspired by a project by @sdatkinson.

Since Tube Amplifiers typically act non-linearly to the input signal, it becomes an interesting problem to model this using a neural network.
The output signal of the amplifier depends on a 'history' of input samples:

y <sub>t</sub> = x_{t}

