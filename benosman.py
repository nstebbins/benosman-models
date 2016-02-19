#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(formatter={'float': '{: 0.2f}'.format})

# global variables
TO_MS = 1000

class synapses(object):

	def __init__(self, v_syn, ge_syn, gf_syn, gate_syn):
		self.v_syn = v_syn
		self.ge_syn = ge_syn
		self.gf_syn = gf_syn
		self.gate_syn = gate_syn

	def print_syn(self):
		for header in ["v_syn", "ge_syn", "gf_syn", "gate_syn"]:
			print header + ": " + np.array_str(self.__dict__[header])

class neuron(object):
	pass

class input_neuron(neuron): # e.g. recall neuron

	def __init__(self, t, spike_prob_thresh = 0): # default constructor
		self.spikes = self.gen_random_spikes(t, 110, spike_prob_thresh)

	def gen_random_spikes(self, t, spike_step_ms, spike_prob_thresh):
		'''spike_step is min interval between spikes (so we can encode)
			spike_prob_thresh is min threshold check (influences spike density)'''

		spikes = np.zeros(np.shape(t))

		for i in xrange(0, np.size(spikes), spike_step_ms * 10 + 1):
			spikes[i] = 1 if np.random.rand(1, 1) > spike_prob_thresh else 0

		return spikes

	def plot_spikes(self, t):
		plt.figure()
		plt.subplot(111)

		plt.bar(t, self.spikes, 5)
		axes = plt.gca()
		axes.set_ylim([0, 2])
		plt.xlabel("time (ms)")
		plt.ylabel("spikes")
		plt.grid(True)
		plt.title("spike times of recall neuron")

		plt.show()

class network_neuron(neuron):

	def __init__(self, v_syn, ge_syn, gf_syn, gate_syn):
		self.syn = synapses(v_syn, ge_syn, gf_syn, gate_syn)

	def gen_voltage(self):
		tau_m = 100 * TO_MS; tau_f = 20
		V_thresh = 10; V_reset = 0

def main():

	# setup

	t = np.multiply(TO_MS, np.arange(0, 1.5, 1e-4)) # time in MS
	print "time: " + np.array_str(t)

	# recall (input) neuron

	recall_neuron = input_neuron(t)
	print "recall spike vector: " + np.array_str(recall_neuron.spikes)

	recall_neuron.plot_spikes(t)

	# output neuron

	output_neuron = network_neuron(np.zeros(np.shape(t)),
		np.zeros(np.shape(t)), np.zeros(np.shape(t)), np.zeros(np.shape(t)))
	output_neuron.syn.print_syn()

	# recall neuron plot output
	recall_neuron.plot_spikes(t)

	# trim, such that minimum spike width is f(1) + eps
	# note: we want no interference; each spike should encode

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
