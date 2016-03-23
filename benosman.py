#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(formatter={'float': '{: 0.2f}'.format})

TO_MS = 1000 # convert to milliseconds

class synapses(object):

    def __init__(self, v_syn, ge_syn, gf_syn, gate_syn):
        self.v_syn = v_syn
        self.ge_syn = ge_syn
        self.gf_syn = gf_syn
        self.gate_syn = gate_syn

    def print_syn(self):
        for header in ["v_syn", "ge_syn", "gf_syn", "gate_syn"]:
            print(header + ": " + np.array_str(self.__dict__[header]))

class neuron(object):

    TIME_POS_CONV = 10 # because time has increments of 0.1

    def __init__(self, t, spikes, values_to_encode = None): # all neurons have spikes + time frame
        self.t = t
        self.spikes = spikes
        self.values_to_encode = values_to_encode

    def encode_values(self): # augmenting self.spikes with encoded values

        T_min = 10; T_cod = 100 # constants

        spike_poses = self.get_spike_poses()

        for i in range(np.size(spike_poses)):
            '''index_f denotes shift amount,
                en_spike_idx denotes encoding spike index'''
            shift_pos_amt = self.TIME_POS_CONV * round(T_min + self.values_to_encode[i] * T_cod, 1)
            en_spike_idx = int(shift_pos_amt + spike_poses[i])

            if en_spike_idx < np.size(self.spikes): # update spike poses
                self.spikes[en_spike_idx] = 1


    def get_spike_poses(self):
        return np.where(self.spikes == 1)[0]

    def plot_spikes(self):
        plt.figure()
        plt.scatter(self.t, self.spikes, marker='None')

        # spike poses
        spike_poses = self.get_spike_poses()
        labels = ['t={0}ms'.format(i / self.TIME_POS_CONV) for i in spike_poses]

        # add special markup for spike times
        for spike_pos in spike_poses:
            plt.scatter(self.t[spike_pos], self.spikes[spike_pos], color='b', marker='o')

        mult = 1 # control for labels
        for label, x in zip(labels, spike_poses):
            plt.annotate( # from stack overflow
                label,
                xy = (x / self.TIME_POS_CONV, 1), xytext = (0, mult * 40),
                textcoords = 'offset points', ha = 'right', va = 'bottom',
                bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
                arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
            mult = mult * -1

        axes = plt.gca()
        axes.set_ylim([0, 2])
        plt.xlabel("time (ms)")
        plt.grid(True)
        plt.title("spike train for neuron")

        plt.show()

class input_neuron(neuron): # e.g. recall neuron

    def __init__(self, t, spike_prob_thresh = 0, values_to_encode = None):
        '''initializes both time and spikes (randomly generated)'''

        self.t = t
        self.gen_random_spikes(spike_prob_thresh)
        self.values_to_encode = values_to_encode
        super(input_neuron, self).__init__(self.t,
            self.spikes, self.values_to_encode)

    def gen_random_spikes(self, spike_prob_thresh):
        '''spike_step is min interval between spikes (so we can encode)
            spike_prob_thresh is min threshold check (influences spike density)'''

        self.spikes = np.zeros(np.shape(self.t))

        for i in range(0, np.size(self.spikes), 110 * super(input_neuron, self).TIME_POS_CONV + 1):
            self.spikes[i] = 1 if np.random.rand(1, 1) > spike_prob_thresh else 0

class network_neuron(neuron):

    def __init__(self, v_syn, ge_syn, gf_syn, gate_syn, t, values_to_encode = None):
        '''initializes time, voltage, and spikes'''

        self.syn = synapses(v_syn, ge_syn, gf_syn, gate_syn)
        self.t = t
        self.gen_voltage() # create spikes
        self.values_to_encode = values_to_encode
        super(network_neuron, self).__init__(self.t, self.spikes,
            self.values_to_encode)

    def gen_voltage(self): # sets spikes and v
        '''model voltage of network neuron'''

        # constants for voltage model
        tau_m = 100 * TO_MS; tau_f = 20
        V_thresh = 10; V_reset = 0
        dt = self.t[1] - self.t[0]
        reset = False

        self.spikes = np.zeros(np.shape(self.t))

        self.v = np.copy(self.syn.v_syn)
        gf = np.copy(self.syn.gf_syn)

        gate = 1 if self.syn.gate_syn[0] == 1 else 0 # default
        kge = self.syn.ge_syn[0]
        ge = 0

        # gate = 1; ge = 300; gf[0] = 40000 #@DEBUG to test model

        for i in range(1, np.size(self.t)):
            # update state variables (biases)
            gate = 1 if self.syn.gate_syn[i] == 1 else 0 if self.syn.gate_syn[i] == -1 else gate
            ge = ge + self.syn.ge_syn[i]

            # reset on gf and voltage
            if reset:
                gf_prev = 0
                v_prev = 0
                reset = False # toggle off until next spike
            else:
                gf_prev = gf[i - 1]
                v_prev = self.v[i - 1]

            # update dynamic conductance (note: gf[i] added in for bias)
            gf[i] = gf_prev + gf[i] + dt * ((-gf[i - 1]) / tau_f)

            # update voltage and check for spike (note: v[i] added in for bias)
            self.v[i] = v_prev + self.v[i] + dt * ((ge + gf[i] * gate) / tau_m)

            # check for reset
            if self.v[i] >= V_thresh:
                reset = True # trigger resets on v and gf
                ge = 0
                gate = 0
                self.spikes[i] = 1 # add spike

def main():

    t = np.multiply(TO_MS, np.arange(0, 1.5, 1e-4)) # time in MS
    V_thresh = 10 # note: as a design consideration, make this member of network neuron

    # RECALL (input neuron)

    recall_neuron = input_neuron(t, 0.7)
    print("INPUT NEURON")

    # BEFORE ENCODING
    print("spike indices (BEFORE encoding):")
    print(recall_neuron.get_spike_poses())
    recall_neuron.plot_spikes()

    recall_neuron.values_to_encode = np.random.uniform(0,
        1, np.size(recall_neuron.get_spike_poses()))
    recall_neuron.encode_values()

    # AFTER ENCODING
    print("spike indices (AFTER encoding):")
    print(recall_neuron.get_spike_poses())
    recall_neuron.plot_spikes()

    # OUTPUT (output neuron)
    output_neuron_v_syn = np.multiply(recall_neuron.spikes, V_thresh)

    output_neuron = network_neuron(output_neuron_v_syn,
       np.zeros(np.shape(t)), np.zeros(np.shape(t)), np.zeros(np.shape(t)), t)


    '''
    output_neuron = network_neuron(np.zeros(np.shape(t)),
        np.zeros(np.shape(t)), np.zeros(np.shape(t)), np.zeros(np.shape(t)), t)
    '''

    print("OUTPUT NEURON")
    # output_neuron.syn.print_syn() # displays synapse components
    # print("voltage" + np.array_str(output_neuron.v))

    print("spike indices:")
    print(list(output_neuron.get_spike_poses()))

    # plot voltage
    plt.figure()
    plt.plot(output_neuron.t, output_neuron.v)
    plt.show()

    # MISCELLANEOUS

    # for each spike, there is also an associated x value
    # note: encode with f(x), denote with random variable?

    # thus, combining encodings with original spikes,
    # you have augmented spike list

    # shift everything by T_syn

    # read in spike train and the corresponding weights to output

    # implement actual neural model for postsynaptic neuron
    # call with w_es, thereby eliciting spikes in postsynaptic neuron

    # display outputs, and decode to obtain original x values

main()
