#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt

# constants (time constants in mS)
TO_MS = 1000; T_TO_POS = 10
T_min = 10; T_cod = 100; T_max = T_min + T_cod # time range
T_syn = 1; T_neu = 0.1 # std. delays (slightly modified T_neu)

tau_m = 100 * TO_MS; tau_f = 20
V_t = 10; V_reset = 0 # voltage model params
w_e = V_t; w_i = -V_t # std. voltage weights
g_mult = V_t * tau_m / tau_f
w_acc = V_t * tau_m / T_max; w_bar_acc = V_t * tau_m / T_cod

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

    def next_v(self, i): # compute voltage at pos i

        # constants (time in mS; V in mV)
        dt = self.t[1] - self.t[0]
        global V_t, V_reset, tau_m, tau_f

        if self.v[i-1] >= V_t:
            v_p = V_reset; ge_p = 0; gf_p = 0; gate_p = 0
        else:
            v_p = self.v[i-1]
            ge_p = self.g_e[i-1]
            gf_p = self.g_f[i-1]
            gate_p = self.gate[i-1]

        self.g_e[i] = self.g_e[i] + ge_p
        self.gate[i] = self.gate[i] + gate_p
        self.g_f[i] = self.g_f[i] + gf_p + dt * (-gf_p / tau_f)
        self.v[i] = self.v[i] + v_p + dt * ((ge_p + gf_p * gate_p) / tau_m)

class adj_matrix(object):

    def __init__(self, neurons, synapses):
        self.neurons = neurons
        self.synapse_matrix = np.empty((neurons.size, neurons.size),
            dtype=object)

        # fill in synapse matrix
        for synapse_list in synapses:
            i = synapse_list.n_from; j = synapse_list.n_to
            self.synapse_matrix[i][j] = synapse_list

    def simulate(self): # update voltages for neurons
        global V_t
        t = (self.neurons[0].t) # retrieve time window
        for tj in range(1, t.size):
            for ni in range(self.neurons.size):
                self.neurons[ni].next_v(tj)
                if self.neurons[ni].v[tj] >= V_t:

                    # for debugging
                    print("spike: (neuron) " + str(ni) + ", (tj) " + str(tj))

                    # check adjacency matrix for synapses to send out
                    for n_to in range(0,self.neurons.size):
                        if self.synapse_matrix[ni][n_to] is not None:
                            for syn in self.synapse_matrix[ni][n_to].synapses:
                                self.synapse_prop(syn, n_to, tj)

    def synapse_prop(self, syn, n_to, tj):
        global T_TO_POS

        # for debugging
        print(str(syn.s_type) + ", " + str(syn.s_weight) + ", " + str(syn.s_delay))

        if syn.s_type is "V":
            self.neurons[n_to].v[tj + int(T_TO_POS * syn.s_delay)] += syn.s_weight
        elif syn.s_type is "g_e":
            self.neurons[n_to].g_e[tj + int(T_TO_POS * syn.s_delay)] += syn.s_weight
        elif syn.s_type is "g_f":
            self.neurons[n_to].g_f[tj + int(T_TO_POS * syn.s_delay)] += syn.s_weight
        else: # gate synapse
            if syn.s_weight is 1:
                self.neurons[n_to].gate[tj + int(T_TO_POS * syn.s_delay)] = 1
            elif syn.s_weight is -1:
                self.neurons[n_to].gate[tj + int(T_TO_POS * syn.s_delay)] = 0
            else:
                pass # throw error

def plot_v(neurons): # can print no more than 9 neurons
        plt.figure()
        for i in range(neurons.size):
            subplot_i = neurons.size * 100 + 10 + i + 1
            plt.subplot(subplot_i)
            plt.plot(neurons[i].t, neurons[i].v)
            plt.xlabel('time (ms)')
            plt.ylabel('voltage (mV)')
        plt.show()

def main(): # logarithm model

    # time frame
    t = np.multiply(TO_MS, np.arange(0, 2, 1e-4)) # time in MS

    # neurons
    input_neuron = neuron(t)
    first_neuron = neuron(t)
    last_neuron = neuron(t)
    acc_neuron = neuron(t)
    output_neuron = neuron(t)

    input_neuron.v[2000] = V_t; input_neuron.v[2700] = V_t

    neurons = np.asarray([input_neuron, first_neuron, last_neuron,
        acc_neuron, output_neuron])

    # synapses
    synapses = np.asarray([
        synapse_list(0, 1, np.asarray([
            synapse("V", w_e, T_syn)
        ])),
        synapse_list(1, 1, np.asarray([
            synapse("V", w_i, T_syn)
        ])),
        synapse_list(0, 2, np.asarray([
            synapse("V", 0.5 * w_e, T_syn)
        ])),
        synapse_list(1, 3, np.asarray([
            synapse("g_e", w_bar_acc, T_syn + T_min)
        ])),
        synapse_list(2, 3, np.asarray([
            synapse("g_e", -w_bar_acc, T_syn),
            synapse("g_f", g_mult, T_syn),
            synapse("gate", 1, T_syn)
        ])),
        synapse_list(2, 4, np.asarray([
            synapse("V", w_e, 2 * T_syn)
        ])),
        synapse_list(3, 4, np.asarray([
            synapse("V", w_e, T_syn + T_min)
        ]))
    ])

    # adjacency matrix
    synapse_matrix = adj_matrix(neurons, synapses)
    synapse_matrix.simulate()

    # outputs
    # for debugging: output to file
    with open("output.txt", "a") as fp:
        fp.truncate(0)
        for i in range(t.size):
            fp.write("(" + str(t[i]) + ", " + str(acc_neuron.v[i])
                 + ", " + str(acc_neuron.g_e[i]) + ", " + str(acc_neuron.g_f[i])
                 + ", " + str(acc_neuron.gate[i]) + ")\n")
    plot_v(neurons) # display voltages



main()
