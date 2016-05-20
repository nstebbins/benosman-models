
import numpy as np
import math
import matplotlib.pyplot as plt

from constants.constants import *
from neural.syn import *
from neural.neuron import *
from neural.predefined_models import *

class adj_matrix(object):

    def __init__(self):
        '''default constructor'''

    def fill_in(self, neurons, synapses, neuron_names):
        '''fill in matrix with synapses'''

        synapse_matrix = np.empty((neurons.size, neurons.size),
            dtype = object)

        # fill in synapse matrix
        for synapse_list in synapses:
            i = neuron_names.index(synapse_list.n_from)
            j = neuron_names.index(synapse_list.n_to)
            synapse_matrix[i][j] = synapse_list

        return(synapse_matrix)

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

def init_neu(neuron_names, t, data = None):
    '''initialize neurons with some data, if necessary'''

    neurons = np.asarray([neuron(label, t) for label in neuron_names])

    # setting stimuli spikes
    if data is not None:
        for key, value in data.items(): # for each neuron
            for j in list(value):
                neurons[neuron_names.index(key)].v[j] = V_t

    return(neurons)

def len_neurons(f_name):
    '''helper method: get number of highest-level neurons in a network'''
    return(len(functions[f_name]["neuron_names"]))


def simulate_neurons(f_name, data = {}):
    '''implementation of a neural model'''

    # time frame
    t = np.multiply(TO_MS, np.arange(0, functions[f_name]["t"], 1e-4)) # time in MS

    # create queue with all network names
    tempq = [(f_name, -1, 0)] # (network name, parent pos, my pos)
    networkq = []

    cumul_tot = len_neurons(f_name)

    while tempq: # visit net and add all subnets
        curr, rootpos, currpos = tempq.pop(0)

        if "subnets" in functions[curr]:
            for subnet in functions[curr]["subnets"]:
                tempq.append((subnet["name"], currpos, cumul_tot))
                cumul_tot += len_neurons(subnet["name"])

        networkq.append((curr, rootpos, currpos))

    print(networkq)

    # create adjacency matrix from queue

    aug_matrix = adj_matrix()
    aug_matrix.synapse_matrix = np.empty((cumul_tot, cumul_tot), dtype = object)

    aug_matrix.neurons = []
    aug_matrix.neuron_names = []

    for net in networkq:
        curr, rootpos, currpos = net

        neu_names = functions[curr]["neuron_names"]
        neurons = init_neu(neu_names, t)

        # fill in portion of synapse matrix
        sub_matrix = aug_matrix.fill_in(neurons, functions[curr]["synapses"], neu_names)

        endpos = currpos + len_neurons(curr)
        aug_matrix.synapse_matrix[currpos:endpos, currpos:endpos] = sub_matrix

        aug_matrix.neuron_names += neu_names
        aug_matrix.neurons += neurons

        print(aug_matrix.synapse_matrix)

    # aug_matrix.simulate() # simulate network

    return((functions[f_name]["output_idx"], neurons))

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
