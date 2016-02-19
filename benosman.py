#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt

# global variables
TO_MS = 1000

class synapses(object):

	def __init__(self, v_syn, ge_syn, gf_syn, gate_syn):
		self.v_syn = v_syn
		self.ge_syn = ge_syn
		self.gf_syn = gf_syn
		self.gate_syn = gate_syn

	def print_syn(self):
		np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
		for header in ["v_syn", "ge_syn", "gf_syn", "gate_syn"]:
			print header + ": " + np.array_str(self.__dict__[header])

class neuron(object):
	pass

class input_neuron(neuron): # e.g. recall neuron

	def __init__(self, t): # default constructor
		self.spikes = self.gen_random_spikes(t)

	def gen_random_spikes(self, t): # possible spike every 110ms
		spikes = np.zeros(np.shape(t))
		return spikes

class network_neuron(neuron):

	def __init__(self, v_syn, ge_syn, gf_syn, gate_syn):
		self.syn = synapses(v_syn, ge_syn, gf_syn, gate_syn)

	def gen_voltage(self):
		tau_m = 100 * TO_MS; tau_f = 20
		V_thresh = 10; V_reset = 0

def main():

	# setup
	t = np.multiply(TO_MS, np.arange(0, 1.5, 1e-5)) # time in MS

	recall_neuron = input_neuron(t)
	print recall_neuron.spikes

	empty_syn = np.zeros(np.shape(t))

	output_neuron = network_neuron(empty_syn, empty_syn, empty_syn, empty_syn)
	output_neuron.syn.print_syn()

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
