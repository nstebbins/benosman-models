#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt

TO_MS = 1000

class synapse(object):

    def __init__(self):
        s_type = None
        s_weight = None

class neuron(object):

    def __init__(self):
        pass

    def next_v(self): # compute next voltage unit
        pass

class adj_matrix(object):

    def __init__(self):
        pass

def main(): # logarithm model

    # time frame
    t = np.multiply(TO_MS, np.arange(0, 1.5, 1e-4)) # time in MS

    # initialize neurons
    input_neuron = neuron()
    first_neuron = neuron()
    last_neuron = neuron()
    acc_neuron = neuron()
    output_neuron = neuron()

    # initialize adj matrix

    # simulate

    # display input spikes

    # display output spikes
