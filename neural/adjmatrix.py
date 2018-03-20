from neural.constants import *

class AdjMatrix(object):

    def __init__(self):
        """default constructor"""

    def fill_in(self, synapses, neuron_names):
        """fill in matrix with synapses"""

        for syn_list in synapses:

            i = neuron_names.index(syn_list.n_from)
            j = neuron_names.index(syn_list.n_to)

            self.synapse_matrix[i][j] = syn_list

    def simulate(self):
        """update voltages for neurons"""

        t = (self.neurons[0].t)  # retrieve time window
        for tj in range(1, t.size):
            for ni in range(len(self.neurons)):
                self.neurons[ni].next_v(tj)
                if self.neurons[ni].v[tj] >= V_t:
                    # check adjacency matrix for synapses to send out
                    for n_to in range(0, len(self.neurons)):
                        if self.synapse_matrix[ni][n_to] is not None:
                            for syn in self.synapse_matrix[ni][n_to].synapses:
                                self.synapse_prop(syn, n_to, tj)

    def synapse_prop(self, syn, n_to, tj):
        """propagate the synapse through the adjacency matrix"""

        t_delay = tj + int(T_TO_POS * (syn.s_delay + T_neu))

        if syn.s_type is "V":
            self.neurons[n_to].v[t_delay] += syn.s_weight
        elif syn.s_type is "g_e":
            self.neurons[n_to].g_e[t_delay] += syn.s_weight
        elif syn.s_type is "g_f":
            self.neurons[n_to].g_f[t_delay] += syn.s_weight
        else:  # gate synapse
            if syn.s_weight is 1:
                self.neurons[n_to].gate[t_delay] = 1
            elif syn.s_weight is -1:
                self.neurons[n_to].gate[t_delay] = 0
            else:
                pass  # throw error
