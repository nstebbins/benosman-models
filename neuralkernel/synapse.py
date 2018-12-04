from collections import namedtuple

Synapse = namedtuple("Synapse", ("type", "weight", "delay"))
SynapseGroup = namedtuple("SynapseGroup", ("src", "dest", "synapses"))
SynapseGroupKey = namedtuple("SynapseGroupKey", ("src", "dest"))
