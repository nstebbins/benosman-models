import numpy as np
import math
import matplotlib.pyplot as plt

# constants (time constants in mS)
TO_MS = 1000; T_TO_POS = 10
T_min = 10; T_cod = 100; T_max = T_min + T_cod # time range
T_syn = 1; T_neu = 0.1 # std. delays (slightly modified T_neu)

tau_m = 100 * TO_MS; tau_f = 20
V_t = 10; V_reset = 0 # voltage model params
w_e = V_t; w_i = -V_t # std. voltage weights
g_mult = V_t * tau_m / tau_f
w_acc = V_t * tau_m / T_max; w_bar_acc = V_t * tau_m / T_cod

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
                (1) within the overall network
                (2) from overall network to subnetwork
                (3) within subnetwork
                (4) from subnetwork to overall network
            - this is used when augmenting the adjacency matrix with subnets
        '''

        if syntype is None:
            self.syntype = 1
        else:
            self.syntype = syntype

class neuron(object):

    def __init__(self, ID, t):
        self.ID = ID
        self.t = t
        self.v = np.zeros(np.shape(t))

        self.g_e = np.zeros(np.shape(t))
        self.g_f = np.zeros(np.shape(t))
        self.gate = np.zeros(np.shape(t))

    def next_v(self, i): # compute voltage at pos i

        # constants (time in mS; V in mV)
        dt = self.t[1] - self.t[0]
        global V_t, V_reset, tau_m, tau_f

        if self.v[i-1] >= V_t:
            v_p = V_reset; ge_p = 0; gf_p = 0; gate_p = 0
        else:
            v_p = self.v[i-1]
            ge_p = self.g_e[i-1]
            gf_p = self.g_f[i-1]
            gate_p = self.gate[i-1]

        self.g_e[i] = self.g_e[i] + ge_p
        self.gate[i] = max([self.gate[i], gate_p])
        self.g_f[i] = self.g_f[i] + gf_p + dt * (-gf_p / tau_f)
        self.v[i] = self.v[i] + v_p + dt * ((ge_p + gf_p * gate_p) / tau_m)

class adj_matrix(object):

    def __init__(self, neurons, synapses, neuron_names):
        self.neurons = neurons
        self.neuron_names = neuron_names
        self.synapse_matrix = np.empty((neurons.size, neurons.size),
            dtype = object)

        # fill in synapse matrix
        for synapse_list in synapses:
            i = self.neuron_names.index(synapse_list.n_from)
            j = self.neuron_names.index(synapse_list.n_to)
            self.synapse_matrix[i][j] = synapse_list

    def simulate(self): # update voltages for neurons
        global V_t
        t = (self.neurons[0].t) # retrieve time window
        for tj in range(1, t.size):
            for ni in range(self.neurons.size):
                self.neurons[ni].next_v(tj)
                if self.neurons[ni].v[tj] >= V_t:

                    # check adjacency matrix for synapses to send out
                    for n_to in range(0,self.neurons.size):
                        if self.synapse_matrix[ni][n_to] is not None:
                            for syn in self.synapse_matrix[ni][n_to].synapses:
                                self.synapse_prop(syn, n_to, tj)

    def synapse_prop(self, syn, n_to, tj):
        global T_TO_POS

        if syn.s_type is "V":
            self.neurons[n_to].v[tj + int(T_TO_POS * syn.s_delay)] += syn.s_weight
        elif syn.s_type is "g_e":
            self.neurons[n_to].g_e[tj + int(T_TO_POS * syn.s_delay)] += syn.s_weight
        elif syn.s_type is "g_f":
            self.neurons[n_to].g_f[tj + int(T_TO_POS * syn.s_delay)] += syn.s_weight
        else: # gate synapse
            if syn.s_weight is 1:
                self.neurons[n_to].gate[tj + int(T_TO_POS * syn.s_delay)] = 1
            elif syn.s_weight is -1:
                self.neurons[n_to].gate[tj + int(T_TO_POS * syn.s_delay)] = 0
            else:
                pass # throw error

def plot_v(neurons):

    f, axarr = plt.subplots(neurons.size, 1, figsize=(15,10), squeeze=False)
    for i in range(neurons.size):
        axarr[i,0].plot(neurons[i].t, neurons[i].v)
        axarr[i,0].set_title('voltage for ' + neurons[i].ID)
    plt.setp([a.get_xticklabels() for a in axarr[:,0]], visible = False)
    plt.setp([axarr[neurons.size - 1,0].get_xticklabels()], visible = True)
    plt.show()

def inspect_neuron(neuron):
    with open("output.txt", "a") as fp: # output to file
        fp.truncate(0)
        for i in range(t.size):
            fp.write("(" + str(t[i]) + ", " + str(acc_neuron.v[i])
                 + ", " + str(acc_neuron.g_e[i]) + ", " + str(acc_neuron.g_f[i])
                 + ", " + str(acc_neuron.gate[i]) + ")\n")

def initialize_neurons(neuron_names, data, t):
    neurons = np.asarray([neuron(label, t) for label in neuron_names])

    # setting stimuli spikes
    for key,value in data.items(): # for each neuron
        for j in list(value):
            neurons[neuron_names.index(key)].v[j] = V_t

    return(neurons)

def simulate_neurons(f_name, data = {}):

    functions = {
        "logarithm" : {
            "t" : 0.5,
            "neuron_names" : ["input", "first", "last", "acc", "output"],
            "synapses" : np.asarray([
                synapse_list("input", "first", np.asarray([
                    synapse("V", w_e, T_syn)
                ])),
                synapse_list("first", "first", np.asarray([
                    synapse("V", w_i, T_syn)
                ])),
                synapse_list("input", "last", np.asarray([
                    synapse("V", 0.5 * w_e, T_syn)
                ])),
                synapse_list("first", "acc", np.asarray([
                    synapse("g_e", w_bar_acc, T_syn + T_min)
                ])),
                synapse_list("last", "acc", np.asarray([
                    synapse("g_e", -w_bar_acc, T_syn),
                    synapse("g_f", g_mult, T_syn),
                    synapse("gate", 1, T_syn)
                ])),
                synapse_list("last", "output", np.asarray([
                    synapse("V", w_e, 2 * T_syn)
                ])),
                synapse_list("acc", "output", np.asarray([
                    synapse("V", w_e, T_syn + T_min)
                ]))
            ]),
            "output_idx" : [4]
        },
        "maximum" : {
            "t" : 1,
            "neuron_names" : ["input", "input2", "larger1", "larger2", "output"],
            "synapses" : np.asarray([
                synapse_list("input", "larger2", np.asarray([
                    synapse("V", 0.5 * w_e, T_syn)
                ])),
                synapse_list("input", "output", np.asarray([
                    synapse("V", 0.5 * w_e, T_syn)
                ])),
                synapse_list("input2", "output", np.asarray([
                    synapse("V", 0.5 * w_e, T_syn)
                ])),
                synapse_list("input2", "larger1", np.asarray([
                    synapse("V", 0.5 * w_e, T_syn + T_min)
                ])),
                synapse_list("larger1", "larger2", np.asarray([
                    synapse("V", w_i, T_syn),
                ])),
                synapse_list("larger2", "larger1", np.asarray([
                    synapse("V", w_i, T_syn)
                ]))
            ]),
            "output_idx" : [4]
        },
        "inverting_memory" : {
            "t" : 0.8,
            "neuron_names" : ["input", "first", "last", "acc",
                "recall", "output"],
            "synapses" : np.asarray([
                synapse_list("input", "first", np.asarray([
                    synapse("V", w_e, T_syn)
                ])),
                synapse_list("first", "first", np.asarray([
                    synapse("V", w_i, T_syn)
                ])),
                synapse_list("input", "last", np.asarray([
                    synapse("V", 0.5 * w_e, T_syn)
                ])),
                synapse_list("first", "acc", np.asarray([
                    synapse("g_e", w_acc, T_syn + T_min)
                ])),
                synapse_list("last", "acc", np.asarray([
                    synapse("g_e", -w_acc, T_syn),
                ])),
                synapse_list("acc", "output", np.asarray([
                    synapse("V", w_e, T_syn)
                ])),
                synapse_list("recall", "acc", np.asarray([
                    synapse("g_e", w_acc, T_syn)
                ])),
                synapse_list("recall", "output", np.asarray([
                    synapse("V", w_e, 2 * T_syn)
                ]))
            ]),
            "output_idx" : [5]
        },
        "non_inverting_memory" : {
            "t" : 0.8,
            "neuron_names" : ["input", "first", "last", "acc", "acc2",
            "recall", "ready", "output"],
            "synapses" : np.asarray([
                synapse_list("input", "first", np.asarray([
                    synapse("V", w_e, T_syn)
                ])),
                synapse_list("first", "first", np.asarray([
                    synapse("V", w_i, T_syn)
                ])),
                synapse_list("input", "last", np.asarray([
                    synapse("V", 0.5 * w_e, T_syn)
                ])),
                synapse_list("first", "acc", np.asarray([
                    synapse("g_e", w_acc, T_syn)
                ])),
                synapse_list("acc", "acc2", np.asarray([
                    synapse("g_e", -w_acc, T_syn)
                ])),
                synapse_list("last", "acc2", np.asarray([
                    synapse("g_e", w_acc, T_syn)
                ])),
                synapse_list("acc", "ready", np.asarray([
                    synapse("V", w_e, T_syn)
                ])),
                synapse_list("recall", "acc2", np.asarray([
                    synapse("g_e", w_acc, T_syn)
                ])),
                synapse_list("recall", "output", np.asarray([
                    synapse("V", w_e, T_syn)
                ])),
                synapse_list("acc2", "output", np.asarray([
                    synapse("V", w_e, T_syn)
                ]))
            ]),
            "output_idx" : [7]
        },
        "synchronizer" : {
            "t" : 1,
            "neuron_names" : ["input0", "input1", "output0", "output1", "sync"],
            "synapses" : np.asarray([]),
            "output_idx" : [2, 3],
            "subnets" : [{
                "name" : "inverting_memory",
                "n" : 2,
                "synapses" : np.asarray([
                    synapse_list("input", "input", np.asarray([
                        synapse("V", w_e, T_syn)
                    ]), 2),
                    synapse_list("output", "output", np.asarray([
                        synapse("V", w_e, T_syn)
                    ]), 4),
                    synapse_list("sync", "recall", np.asarray([
                        synapse("V", w_e, T_syn)
                    ]), 2),
                    synapse_list("ready", "sync", np.asarray([
                        synapse("V", 0.5 * w_e, T_syn)
                    ]), 4)
                ])
            }]
        }
    }

    f_p = functions[f_name] # parameters

    # time frame & neurons
    t = np.multiply(TO_MS, np.arange(0, f_p["t"], 1e-4)) # time in MS
    neurons = initialize_neurons(
        f_p["neuron_names"], data, t)

    # adjacency matrix
    syn_matrix = adj_matrix(neurons, f_p["synapses"], f_p["neuron_names"])

    # handle subnets (augment adjacency matrix)
    if "subnets" in f_p:
        print("[subnets present]...")
        for subnet_type in f_p["subnets"]: # each network type
            print("[each subnet]...")
            subnet_neuron_names = functions[subnet_type["name"]]["neuron_names"]
            to_add = len(subnet_neuron_names)

            for subnet in range(subnet_type["n"]): # each network for the network type

                offset = int(math.sqrt((syn_matrix.synapse_matrix).size))
                aug_dim = to_add + offset

                # initialize augmented matrix
                aug_matrix = np.empty((aug_dim, aug_dim), dtype = object)
                aug_matrix[:offset, :offset] = syn_matrix.synapse_matrix

                # iterate over synapses & add them to the preexisting adjacency matrix
                for synlist in subnet_type["synapses"]:
                    if synlist.syntype == 1:
                        pass
                    elif synlist.syntype == 2:
                        i = f_p["neuron_names"].index(synapse_list.n_from + str(subnet))
                        j = offset + subnet_neuron_names.index(synapse_list.n_to)
                    elif synlist.syntype == 3:
                        i = offset + subnet_neuron_names.index(synapse_list.n_from)
                        j = offset + subnet_neuron_names.index(synapse_list.n_to)
                    else:
                        i = offset + subnet_neuron_names.index(synapse_list.n_from)
                        j = f_p["neuron_names"].index(synapse_list.n_to + str(subnet))

                    aug_matrix[i][j] = synlist

                syn_matrix.synapse_matrix = aug_matrix

    syn_matrix.simulate()

    return((f_p["output_idx"], neurons))
