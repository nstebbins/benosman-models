class Synapse(object):

    def __init__(self, s_type, s_weight, s_delay):
        self.s_type = s_type
        self.s_weight = s_weight
        self.s_delay = s_delay


class SynapseList(object):

    def __init__(self, n_from, n_to, synapses, synapse_type=None):
        self.n_from = n_from
        self.n_to = n_to
        self.synapses = synapses

        '''
            synapse_type //
            - specifies whether a connection is
                (1) within network
                (2) from network to child network
                (3) from child network to network
            - used when augmenting the adjacency matrix with subnets
        '''

        if synapse_type is None:
            self.synapse_type = 1
        else:
            self.synapse_type = synapse_type
