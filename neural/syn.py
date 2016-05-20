
class synapse(object):

    def __init__(self, s_type, s_weight, s_delay):
        self.s_type = s_type
        self.s_weight = s_weight
        self.s_delay = s_delay

class synapse_list(object):

    def __init__(self, n_from, n_to, synapses, syntype = None):
        self.n_from = n_from
        self.n_to = n_to
        self.synapses = synapses

        '''
            syntype //
            - specifies whether a connection is
                (1) within network
                (2) from network to parent network
                (3) from parent network to network
            - used when augmenting the adjacency matrix with subnets
        '''

        if syntype is None:
            self.syntype = 1
        else:
            self.syntype = syntype
