class Synapse(object):

    def __init__(self, s_type, s_weight, s_delay):
        self.s_type = s_type
        self.s_weight = s_weight
        self.s_delay = s_delay


class SynapseList(object):

    def __init__(self, n_from, n_to, synapses):
        self.n_from = n_from
        self.n_to = n_to
        self.synapses = synapses
