#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt

# constants (time constants in mS)
TO_MS = 1000
V_t = 10; V_reset = 0 # voltage model params
w_e = V_t; w_i = -V_t # standardized synapse weights
T_syn = 1; T_neu = 0.01 # standardized delays

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
        self.t = t
        self.v = np.zeros(np.shape(t))

        self.g_e = np.zeros(np.shape(t))
        self.g_f = np.zeros(np.shape(t))
        self.gate = np.zeros(np.shape(t))

    def plot_v(self): #TBD
        pass

    def plot_spikes(self): #TBD
        pass

    def next_v(self, i): # compute voltage at pos i

        # constants (time consts in mS; V consts in mV)
        tau_m = 100 * TO_MS; tau_f = 20
        dt = self.t[1] - self.t[0]
        global V_t, V_reset

        if self.v[i-1] >= V_t:
            v_p = V_reset; ge_p = 0; gf_p = 0; gate_p = 0
        else:
            v_p = self.v[i-1]
            ge_p = self.g_e[i-1]
            gf_p = self.g_f[i-1]
            gate_p = self.gate[i-1]

        self.g_f[i] = self.g_f[i] + gf_p + dt * (-gf_p / tau_f)
        self.v[i] = self.v[i] + v_p + dt * ((ge_p + gf_p * gate_p) / tau_m)

class adj_matrix(object):

    def __init__(self, neurons, synapses):
        self.neurons = neurons
        self.synapse_matrix = np.empty((neurons.size, neurons.size),
            dtype=object)

        for synapse_list in synapses:
            i = synapse_list.n_from; j = synapse_list.n_to
            self.synapse_matrix[i][j] = synapse_list

        print(np.array_str(self.synapse_matrix))

    def simulate(self): # update voltages for neurons
        global V_t
        t = (self.neurons[0].t) # retrieve time window
        for tj in range(1, t.size):
            for ni in range(self.neurons.size):
                self.neurons[ni].next_v(tj)
                if self.neurons[ni].v[tj] >= V_t:

                    # check adjacency matrix for synapses to send out
                    for n_to in range(0,self.neurons.size):
                        if self.synapse_matrix[ni][n_to] is not None:
                            for syn in self.synapse_matrix[ni][n_to].synapses:
                                print("synapse")

                    # for debugging
                    print("spike: (neuron) " + str(ni) + ", (tj) " + str(tj))


def main(): # logarithm model

    # time frame
    t = np.multiply(TO_MS, np.arange(0, 1.5, 1e-4)) # time in MS

    # neurons
    input_neuron = neuron(t)
    first_neuron = neuron(t)
    last_neuron = neuron(t)
    acc_neuron = neuron(t)
    output_neuron = neuron(t)

    input_neuron.v[400] = V_t; input_neuron.v[800] = V_t

    neurons = np.asarray([input_neuron, first_neuron, last_neuron,
        acc_neuron, output_neuron])

    # synapses
    synapses = np.asarray([
        synapse_list(0, 1, np.asarray([
            synapse("V", w_e, T_syn)
        ]))
    ])

    # adjacency matrix
    synapse_matrix = adj_matrix(neurons, synapses)
    synapse_matrix.simulate()

    # display input spikes

    # display output spikes

main()
