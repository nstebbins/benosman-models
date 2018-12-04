import numpy as np

from .adjmatrix import AdjMatrix
from .constants import TO_MS
from .networks import logarithm
from .neuron import Neuron


def simulate_neurons(network_name, offsets):
    # highest level function, TODO: docstring

    network = logarithm  # TODO: make dynamic with network_name

    # TODO: change this
    t = np.multiply(TO_MS,
                    np.arange(0, network.window, 1e-4))  # time in MS

    neurons = {neuron_name: Neuron(neuron_name, t) for neuron_name in
               network.neuron_names}

    # TODO: improve and put somewhere else
    for neuron_name in offsets:
        neurons[neuron_name].set_spikes(offsets[neuron_name])

    adj_matrix = AdjMatrix(neurons, network.synapses)
    adj_matrix.simulate(t)

    # TODO:
    output_neurons = {}
    for neuron in adj_matrix.neurons:
        if neuron in network.output_idx:
            # copy the reference
            output_neurons[neuron] = adj_matrix.neurons[neuron]

    return output_neurons
