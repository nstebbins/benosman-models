from collections import namedtuple

from .constants import T_NEU

Synapse = namedtuple("Synapse", ("type", "weight", "delay"))
SynapseGroup = namedtuple("SynapseGroup", ("src", "dest", "synapses"))
SynapseGroupKey = namedtuple("SynapseGroupKey", ("src", "dest"))


class SynapseMatrix:

    # TODO: we don't need state
    def __init__(self, neurons, synapses):
        self.neurons = neurons
        self.synapses = synapses

    def simulate(self, window):
        """simulate neurons in the adjacency matrix"""

        for time in window:
            for _, src in self.neurons.items():
                src.simulate(time)
                if src.is_spike(time):
                    # propagate across all destination neurons
                    for _, dest in self.neurons.items():
                        self.propagate_synapse(src, dest, time)

    def propagate_synapse(self, src, dest, time):
        # TODO: docstring
        synapse_key = SynapseGroupKey(src.name, dest.name)
        if synapse_key in self.synapses:
            for synapse in self.synapses[synapse_key]:
                # propagate
                idx = time + synapse.delay + T_NEU
                self.neurons[dest.name].update(synapse.type, idx,
                                               synapse.weight)
