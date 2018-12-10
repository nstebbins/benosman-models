import numpy as np

from .networks import logarithm
from .neuron import Neuron
from .synapse import SynapseMatrix


def simulate_neurons(network_name, offsets):
    # highest level function, TODO: docstring

    network = logarithm  # TODO: make dynamic with network_name
    t = np.arange(network.window)

    neurons = {neuron_name: Neuron(neuron_name, t) for neuron_name in
               network.neuron_names}

    # TODO: improve and put somewhere else
    for neuron_name in offsets:
        neurons[neuron_name].set_spikes(offsets[neuron_name])

    adj_matrix = SynapseMatrix(neurons, network.synapses)
    adj_matrix.simulate(t)

    # TODO: put this somewhere else?
    output_neurons = {}
    for neuron in adj_matrix.neurons:
        if neuron in network.output_neuron_names:
            # copy the reference
            output_neurons[neuron] = adj_matrix.neurons[neuron]

    return output_neurons
