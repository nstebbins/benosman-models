from collections import namedtuple

Synapse = namedtuple("Synapse", ("type", "weight", "delay"))
SynapseList = namedtuple("SynapseList", ("src", "dest", "synapses"))
