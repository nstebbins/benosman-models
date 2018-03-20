import matplotlib.pyplot as plt
import numpy as np

from .adjmatrix import AdjMatrix
from .neuron import Neuron
from .predefined_models import functions
from .constants import V_t, TO_MS


# TODO: remove warning when running this
def plot_v(neurons: np.ndarray) -> None:
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
        ax.set_title('voltage for ' + neurons[i].name)
        start, end = ax.get_xlim()
        ax.xaxis.set_ticks(np.arange(start, end, 100))
        ax.grid(True)

    plt.show()


# TODO: probably put this somewhere else
def initialize_neurons(neuron_names: list, t: np.ndarray, data: dict = None) -> list:
    """initialize neurons with some data, if necessary"""

    neurons = [Neuron(neuron_name, t) for neuron_name in neuron_names]

    # setting stimuli spikes
    if data is not None:
        for key, value in data.items():  # for each neuron
            for j in list(value):
                neurons[neuron_names.index(key)].v[j] = V_t

    return neurons


def simulate_neurons(f_name: str, data: dict) -> (list, list):
    """implementation of a neural model"""

    print("f_name: " + f_name)
    print("data" + str(data))

    # time frame
    t = np.multiply(TO_MS, np.arange(0, functions[f_name]["t"], 1e-4))  # time in MS

    # adjacency matrix
    neurons = initialize_neurons(functions[f_name]["neuron_names"], t, data)
    synapses = functions[f_name]["synapses"]

    adj_matrix = AdjMatrix(neurons, synapses)
    adj_matrix.simulate()

    return functions[f_name]["output_idx"], adj_matrix.neurons
