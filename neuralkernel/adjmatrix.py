from .constants import T_TO_POS, T_NEU
from .synapse import SynapseMatrixKey


class AdjMatrix:

    def __init__(self, neurons, synapses):
        self.neurons = neurons
        self.synapses = synapses

    def simulate(self, window):
        """simulate neurons in the adjacency matrix"""

        for window_idx in range(window.size):
            for _, src in self.neurons.items():
                src.simulate(window_idx)
                if src.is_spike(window_idx):
                    # propagate across all destination neurons
                    for _, dest in self.neurons.items():
                        self.propagate_synapse(src, dest, window_idx)

    def propagate_synapse(self, src, dest, window_idx):
        # TODO: docstring
        synapse_key = SynapseMatrixKey(src.name, dest.name)
        if synapse_key in self.synapses:
            for synapse in self.synapses[synapse_key]:
                print(synapse)
                # propagate
                idx = window_idx + int(T_TO_POS * (synapse.delay + T_NEU))
                self.neurons[dest.name].update(synapse.type, idx,
                                               synapse.weight)
