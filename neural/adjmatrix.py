from constants.constants import *


class AdjMatrix(object):

    def __init__(self):
        '''default constructor'''

    def fill_in(self, synapses, neuron_names, curr_o, child_o):
        '''fill in matrix with synapses'''

        for syn_list in synapses:

            if syn_list.syntype is 1:  # net -> net
                off1 = curr_o;
                off2 = curr_o
            elif syn_list.syntype is 2:  # net -> parent
                off1 = curr_o;
                off2 = child_o
            else:  # parent -> net
                off1 = child_o;
                off2 = curr_o

            i = off1 + neuron_names[off1:].index(syn_list.n_from)
            j = off2 + neuron_names[off2:].index(syn_list.n_to)

            self.synapse_matrix[i][j] = syn_list

    def simulate(self):
        '''update voltages for neurons'''

        global V_t
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
        '''propagate the synapse through the adjacency matrix'''

        global T_TO_POS
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
