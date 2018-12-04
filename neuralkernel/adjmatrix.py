import numpy as np

from .constants import T_TO_POS, T_NEU, V_THRESHOLD


class AdjMatrix:

    def __init__(self, neurons, synapses):
        self.neurons = neurons
        # TODO: clean-up code below
        self.synapse_matrix = np.empty((len(self.neurons), len(self.neurons)),
                                       dtype=object)
        neuron_names = [neuron.name for neuron in self.neurons]
        for synapse_list in synapses:
            i = neuron_names.index(synapse_list.n_from)
            j = neuron_names.index(synapse_list.n_to)
            self.synapse_matrix[i][j] = synapse_list

    def simulate_v2(self, window):
        """simulate neurons in the adjacency matrix"""

        for window_idx in range(window.size):
            for neuron_idx, neuron in enumerate(self.neurons):
                neuron.simulate(window_idx)
                if neuron.is_spike(window_idx):
                    # check adj matrix for synapses to send
                    pass

    def simulate(self):
        """simulate neurons in the adjacency matrix"""

        t = self.neurons[0].t  # retrieve time window
        for tj in range(0, t.size):
            for ni in range(len(self.neurons)):
                self.neurons[ni].simulate(tj)
                if self.neurons[ni].v[tj] >= V_THRESHOLD:
                    # check adjacency matrix for synapses to send out
                    for n_to in range(0, len(self.neurons)):
                        if self.synapse_matrix[ni][n_to]:
                            for syn in self.synapse_matrix[ni][n_to].synapses:
                                print(syn)
                                self.synapse_prop(syn, n_to, tj)

    def synapse_prop(self, syn, n_to, tj):
        """propagate the synapse through the adjacency matrix"""
        t_delay = tj + int(T_TO_POS * (syn.delay + T_NEU))
        self.neurons[n_to].update(syn.type, t_delay, syn.weight)
