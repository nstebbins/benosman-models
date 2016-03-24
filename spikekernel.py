#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt

# constants (time constants in mS)
TO_MS = 1000
V_t = 10; V_reset = 0
w_e = V_t; w_i = -V_t
T_syn = 1; T_neu = 0.01

class synapse(object):

    def __init__(self, s_type, s_weight, s_delay):
        self.s_type = s_type
        self.s_weight = s_weight
        self.s_delay = s_delay

class synapse_list(object):

    def __init__(self, n_from, n_to, synapses):
        self.n_from = n_from
        self.n_to = n_to
        self.synapses = synapses

class neuron(object):

    def __init__(self, t):
        self.v = np.zeros(np.shape(t))

    def next_v(self): # compute next voltage unit

        # constants (time consts in mS; V consts in mV)
        tau_m = 100 * TO_MS; tau_f = 20
        global V_t, V_reset

class adj_matrix(object):

    def __init__(self, neurons, synapses):
        self.synapse_matrix = np.empty((neurons.size, neurons.size), dtype=synapse_list)
        print(np.array_str(self.synapse_matrix))

def main(): # logarithm model

    # time frame
    t = np.multiply(TO_MS, np.arange(0, 1.5, 1e-4)) # time in MS

    # initialize neurons
    input_neuron = neuron(t)
    first_neuron = neuron(t)
    last_neuron = neuron(t)
    acc_neuron = neuron(t)
    output_neuron = neuron(t)

    neurons = np.asarray([input_neuron, first_neuron, last_neuron,
        acc_neuron, output_neuron])

    # initialize adj matrix
    synapses = np.asarray([
        synapse_list(0, 2, np.asarray(synapse("V", w_e, T_syn)))
    ])

    synapse_matrix = adj_matrix(neurons, synapses)

    # simulate

    # display input spikes

    # display output spikes

main()
