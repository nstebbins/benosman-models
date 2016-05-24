
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

    def fill_in(self, synapses, neuron_names, curr_o, child_o):
        '''fill in matrix with synapses'''

        for syn_list in synapses:

            if syn_list.syntype is 1: # net -> net
                off1 = curr_o; off2 = curr_o
            elif syn_list.syntype is 2: # net -> parent
                off1 = curr_o; off2 = child_o
            else: # parent -> net
                off1 = child_o; off2 = curr_o

            i = off1 + neuron_names[off1:].index(syn_list.n_from)
            j = off2 + neuron_names[off2:].index(syn_list.n_to)

            self.synapse_matrix[i][j] = syn_list

    def simulate(self):
        '''update voltages for neurons'''

        global V_t
        t = (self.neurons[0].t) # retrieve time window
        for tj in range(1, t.size):
            for ni in range(len(self.neurons)):
                self.neurons[ni].next_v(tj)
                if self.neurons[ni].v[tj] >= V_t:
                    # check adjacency matrix for synapses to send out
                    for n_to in range(0, len(self.neurons)):
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

    neurons = [neuron(label, t) for label in neuron_names]

    # setting stimuli spikes
    if data is not None:
        for key, value in data.items(): # for each neuron
            for j in list(value):
                neurons[neuron_names.index(key)].v[j] = V_t

    return(neurons)

def len_neurons(f_name):
    '''helper method: get # of highest-level neurons in a network'''

    return(len(functions[f_name]["neuron_names"]))

def get_par_pos(augq, rootpos):
    '''helper method: get index of parent'''

    for i, elem in enumerate(augq):
        if elem[2] is rootpos: # 2 = currpos
            return(i)
    return(-1) # error

def simulate_neurons(f_name, data = {}):
    '''implementation of a neural model'''

    # time frame
    t = np.multiply(TO_MS, np.arange(0, functions[f_name]["t"], 1e-4)) # time in MS

    # ** create queue with all network names

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

    # print(networkq) # COMPLETED NETWORKQ

    # ** initialize adjacency matrix from queue

    aug_matrix = adj_matrix()
    aug_matrix.synapse_matrix = np.empty((cumul_tot, cumul_tot), dtype = object)

    aug_matrix.neuron_names = []

    for net in networkq:
        curr, rootpos, currpos = net
        aug_matrix.neuron_names += functions[curr]["neuron_names"]

    aug_matrix.neurons = init_neu(aug_matrix.neuron_names, t, data)

    # ** augment networkq so that it points forward (to children)

    augq = [[curr, rootpos, currpos, []]
        for curr, rootpos, currpos in networkq]

    for i, net in enumerate(reversed(augq)):
        _, rootpos, _, _ = net
        if rootpos is not -1:
            augq[get_par_pos(augq, rootpos)][3].insert(0, len(augq) - (i + 1)) # 3 = children list

    # print(augq) # COMPLETED AUGQ

    # ** add synapses

    for net in augq:
        curr, rootpos, currpos, childposes = net

        # fill in portion of synapse matrix
        aug_matrix.fill_in(functions[curr]["synapses"],
            aug_matrix.neuron_names, currpos, -1)

        # add connections to children
        if "subnets" in functions[curr]:
            for i, subnet in enumerate(functions[curr]["subnets"]):
                child = augq[childposes[i]]
                aug_matrix.fill_in(subnet["synapses"],
                    aug_matrix.neuron_names, currpos, child[2]) # 2 = currpos

    # print(aug_matrix.synapse_matrix) # COMPLETED SYNMATRIX

    aug_matrix.simulate() # simulate network

    return((functions[f_name]["output_idx"], aug_matrix.neurons))
