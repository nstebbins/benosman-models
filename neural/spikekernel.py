import numpy as np
import matplotlib.pyplot as plt

from neural.neuron import *
from neural.predefined_models import *
from neural.adjmatrix import *


def plot_v(neurons):
    fig = plt.figure(1, figsize=(15, 10), facecolor='white')

    big_ax = fig.add_subplot(111)  # overarching subplot

    # Turn off axis lines and ticks of the big subplot
    big_ax.spines['top'].set_color('none')
    big_ax.spines['bottom'].set_color('none')
    big_ax.spines['left'].set_color('none')
    big_ax.spines['right'].set_color('none')
    big_ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')

    big_ax.set_xlabel('time (ms)')
    big_ax.set_ylabel('voltage (mV)')

    for i in range(neurons.size):
        num = int(str(neurons.size) + "1" + str(i + 1))

        ax = fig.add_subplot(num)
        ax.plot(neurons[i].t, neurons[i].v)
        ax.set_title('voltage for ' + neurons[i].ID)
        start, end = ax.get_xlim()
        ax.xaxis.set_ticks(np.arange(start, end, 100))
        ax.grid(True)

    plt.show()


def init_neu(neuron_names, t, data=None):
    """initialize neurons with some data, if necessary"""

    neurons = [Neuron(label, t) for label in neuron_names]

    # setting stimuli spikes
    if data is not None:
        for key, value in data.items():  # for each neuron
            for j in list(value):
                neurons[neuron_names.index(key)].v[j] = V_t

    return neurons


def len_neurons(f_name):
    """helper method: get # of highest-level neurons in a network"""

    return len(functions[f_name]["neuron_names"])


def get_par_pos(augq, rootpos):
    """helper method: get index of parent"""

    for i, elem in enumerate(augq):
        if elem[2] is rootpos:  # 2 = currpos
            return i
    return -1  # error


def simulate_neurons(f_name, data={}):
    """implementation of a neural model"""

    print("f_name: " + f_name)
    print("data")
    print(data)

    # time frame
    t = np.multiply(TO_MS, np.arange(0, functions[f_name]["t"], 1e-4))  # time in MS

    # adjacency matrix

    adj_matrix = AdjMatrix()
    len_neu = len_neurons(f_name)
    adj_matrix.synapse_matrix = np.empty((len_neu, len_neu), dtype=object)
    adj_matrix.neuron_names = functions[f_name]["neuron_names"]
    adj_matrix.neurons = init_neu(adj_matrix.neuron_names, t, data)

    # synapse matrix
    adj_matrix.fill_in(functions[f_name]["synapses"], adj_matrix.neuron_names)
    adj_matrix.simulate()

    return functions[f_name]["output_idx"], adj_matrix.neurons
