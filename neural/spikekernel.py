
import numpy as np
import matplotlib.pyplot as plt

from neural.neuron import *
from neural.predefined_models import *
from neural.adjmatrix import *

def plot_v(neurons):

    fig = plt.figure(1, figsize = (15, 10), facecolor = 'white')

    big_ax = fig.add_subplot(111) # overarching subplot

    # Turn off axis lines and ticks of the big subplot
    big_ax.spines['top'].set_color('none')
    big_ax.spines['bottom'].set_color('none')
    big_ax.spines['left'].set_color('none')
    big_ax.spines['right'].set_color('none')
    big_ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')

    big_ax.set_xlabel('time (ms)')
    big_ax.set_ylabel('voltage (mV)')

    for i in range(neurons.size):
        num = int(str(neurons.size) + "1" + str(i+1))

        ax = fig.add_subplot(num)
        ax.plot(neurons[i].t, neurons[i].v)
        ax.set_title('voltage for ' + neurons[i].ID)
        start, end = ax.get_xlim()
        ax.xaxis.set_ticks(np.arange(start, end, 100))
        ax.grid(True)

    plt.show()

def init_neu(neuron_names, t, data = None):
    '''initialize neurons with some data, if necessary'''

    neurons = [Neuron(label, t) for label in neuron_names]

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

    aug_matrix = AdjMatrix()
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
