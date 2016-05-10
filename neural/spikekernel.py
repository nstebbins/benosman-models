
import numpy as np
import math
import matplotlib.pyplot as plt

from constants.constants import *
from neural.syn import *
from neural.neuron import *
from neural.predefined_models import *

class adj_matrix(object):

    def __init__(self, neurons, synapses, neuron_names):
        self.neurons = neurons
        self.neuron_names = neuron_names
        self.synapse_matrix = np.empty((neurons.size, neurons.size),
            dtype = object)

        # fill in synapse matrix
        for synapse_list in synapses:
            i = self.neuron_names.index(synapse_list.n_from)
            j = self.neuron_names.index(synapse_list.n_to)
            self.synapse_matrix[i][j] = synapse_list

    def simulate(self):
        '''update voltages for neurons'''

        global V_t
        t = (self.neurons[0].t) # retrieve time window
        for tj in range(1, t.size):
            for ni in range(self.neurons.size):
                self.neurons[ni].next_v(tj)
                if self.neurons[ni].v[tj] >= V_t:
                    # check adjacency matrix for synapses to send out
                    for n_to in range(0, self.neurons.size):
                        if self.synapse_matrix[ni][n_to] is not None:
                            for syn in self.synapse_matrix[ni][n_to].synapses:
                                self.synapse_prop(syn, n_to, tj)

    def synapse_prop(self, syn, n_to, tj):
        '''propagate the synapse through the adjacency matrix'''

        global T_TO_POS
        t_delay = tj + int(T_TO_POS * (syn.s_delay + T_neu))

        if syn.s_type is "V":
            self.neurons[n_to].v[t_delay] += syn.s_weight
        elif syn.s_type is "g_e":
            self.neurons[n_to].g_e[t_delay] += syn.s_weight
        elif syn.s_type is "g_f":
            self.neurons[n_to].g_f[t_delay] += syn.s_weight
        else: # gate synapse
            if syn.s_weight is 1:
                self.neurons[n_to].gate[t_delay] = 1
            elif syn.s_weight is -1:
                self.neurons[n_to].gate[t_delay] = 0
            else:
                pass # throw error

def plot_v(neurons):

    fig = plt.figure(1, figsize=(15, 10))

    for i in range(neurons.size):
        num = int(str(neurons.size) + "1" + str(i+1))

        ax = fig.add_subplot(num)
        ax.plot(neurons[i].t, neurons[i].v)
        ax.set_xlabel('time (ms)')
        ax.set_ylabel('voltage (mV)')
        ax.set_title('voltage for ' + neurons[i].ID)

    plt.show()

def initialize_neurons(neuron_names, t, data = None):
    '''initialize neurons with some data, if necessary'''

    neurons = np.asarray([neuron(label, t) for label in neuron_names])

    # setting stimuli spikes
    if data is not None:
        for key, value in data.items(): # for each neuron
            for j in list(value):
                neurons[neuron_names.index(key)].v[j] = V_t

    return(neurons)

def simulate_neurons(f_name, data = {}):
    '''implementation of a neural model'''

    f_p = functions[f_name] # parameters

    # time frame & neurons
    t = np.multiply(TO_MS, np.arange(0, f_p["t"], 1e-4)) # time in MS
    neurons = initialize_neurons(
        f_p["neuron_names"], t, data)

    # initial adjacency matrix (without subnets)
    syn_matrix = adj_matrix(neurons, f_p["synapses"], f_p["neuron_names"])

    # augment_matrix(syn_matrix, f_p)

    syn_matrix.simulate()

    return((f_p["output_idx"], neurons))

def augment_matrix(syn_matrix, func):
    '''look at curr subnet of interest and add stuff to network'''
    '''NEEDS TO BE TWEAKED'''

    if "subnets" not in func:
        return syn_matrix
    else:
        for subnet in func["subnets"]:

            sub_names = subnet["neuron_names"]

            pre_sze = (syn_matrix.synapse_matrix).shape[0]
            new_sze = len(sub_names) + curr_offset

            sub_neurons = initialize_neurons(sub_names, t)

            syn_matrix.neurons = np.concatenate((neurons, sub_neurons), axis = 0)
            syn_matrix.neuron_names += sub_names

            # sub matrix
            sub_matrix = adj_matrix(sub_neurons,
                functions[subnet["name"]]["synapses"],
                functions[subnet["name"]]["neuron_names"]
            )

            # augmented matrix
            aug_matrix = np.empty((new_sze, new_sze), dtype = object)
            aug_matrix[:pre_sze, :pre_sze] = syn_matrix.synapse_matrix
            aug_matrix[pre_sze:, pre_sze:] = sub_matrix.synapse_matrix

            for synlist in subnet["synapses"]:
                i = syn_matrix.neuron_names.index(synlist.n_from)
                j = syn_matrix.neuron_names.index(synlist.n_to)
                aug_matrix[i][j] = synlist

            syn_matrix.synapse_matrix = aug_matrix
