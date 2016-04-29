
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
                                self.synapse_prop(syn, n_to, tj)

    def synapse_prop(self, syn, n_to, tj):
        global T_TO_POS

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

def plot_v(neurons):

    f, axarr = plt.subplots(neurons.size, 1, figsize=(15,10), squeeze=False)
    for i in range(neurons.size):
        axarr[i,0].plot(neurons[i].t, neurons[i].v)
        axarr[i,0].set_title('voltage for ' + neurons[i].ID)
    plt.setp([a.get_xticklabels() for a in axarr[:,0]], visible = False)
    plt.setp([axarr[neurons.size - 1,0].get_xticklabels()], visible = True)
    plt.show()

def inspect_neuron(neuron):
    with open("output.txt", "a") as fp: # output to file
        fp.truncate(0)
        for i in range(t.size):
            fp.write("(" + str(t[i]) + ", " + str(acc_neuron.v[i])
                 + ", " + str(acc_neuron.g_e[i]) + ", " + str(acc_neuron.g_f[i])
                 + ", " + str(acc_neuron.gate[i]) + ")\n")

def initialize_neurons(neuron_names, t, data = None):
    neurons = np.asarray([neuron(label, t) for label in neuron_names])

    # setting stimuli spikes
    if data is not None:
        for key, value in data.items(): # for each neuron
            for j in list(value):
                neurons[neuron_names.index(key)].v[j] = V_t

    return(neurons)

def simulate_neurons(f_name, data = {}):

    f_p = functions[f_name] # parameters

    # time frame & neurons
    t = np.multiply(TO_MS, np.arange(0, f_p["t"], 1e-4)) # time in MS
    neurons = initialize_neurons(
        f_p["neuron_names"], t, data)

    # initial adjacency matrix (without subnets)
    syn_matrix = adj_matrix(neurons, f_p["synapses"], f_p["neuron_names"])

    # augmented adjacency matrix (with subnets)
    if "subnets" in f_p:
        for subnet_type in f_p["subnets"]: # each network type
            subnet_names = functions[subnet_type["name"]]["neuron_names"]
            to_add = len(subnet_names)

            for subnet in range(subnet_type["n"]): # each network for the network type

                uniq_subnet_names = ["_sub" + subnet_type["name"] +
                    subnet_name + str(subnet) for subnet_name in subnet_names]

                offset = int(math.sqrt((syn_matrix.synapse_matrix).size))
                aug_dim = to_add + offset

                # neurons
                subnet_neurons = initialize_neurons(
                    uniq_subnet_names, t)
                neurons = np.concatenate((neurons, subnet_neurons), axis = 0)

                # get adjacency matrix for the subnet
                sub_matrix = adj_matrix(subnet_neurons,
                    functions[subnet_type["name"]]["synapses"],
                    functions[subnet_type["name"]]["neuron_names"])

                # initialize augmented matrix
                aug_matrix = np.empty((aug_dim, aug_dim), dtype = object)
                aug_matrix[:offset, :offset] = syn_matrix.synapse_matrix
                aug_matrix[offset:, offset:] = sub_matrix.synapse_matrix

                # iterate over synapses & add them to preexisting adj matrix
                for synlist in subnet_type["synapses"]:
                    if synlist.syntype == 1:
                        pass
                    elif synlist.syntype == 2: # overall -> subnet
                        if (synlist.n_from).startswith("_"):
                            i = f_p["neuron_names"].index(
                                    synlist.n_from) # e.g. _sync
                        else:
                            i = f_p["neuron_names"].index(
                                    synlist.n_from + str(subnet))
                        j = offset + subnet_names.index(synlist.n_to)
                    elif synlist.syntype == 3: # subnet -> subnet
                        i = offset + subnet_names.index(synlist.n_from)
                        j = offset + subnet_names.index(synlist.n_to)
                    else: # subnet -> overall
                        i = offset + subnet_names.index(synlist.n_from)
                        if (synlist.n_to).startswith("_"):
                            j = f_p["neuron_names"].index(
                                    synlist.n_to) # e.g. _sync
                        else:
                            j = f_p["neuron_names"].index(
                                    synlist.n_to + str(subnet))

                    aug_matrix[i][j] = synlist

                syn_matrix.neurons = neurons
                syn_matrix.neuron_names += uniq_subnet_names
                syn_matrix.synapse_matrix = aug_matrix

    syn_matrix.simulate()

    return((f_p["output_idx"], neurons))
