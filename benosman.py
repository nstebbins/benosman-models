#!/usr/bin/python

import numpy as np

class synapses(object):

	v_syn, ge_syn, gf_syn, gate_syn = [np.zeros(1) for _ in xrange(4)]

	def __init__(self, v_syn, ge_syn, gf_syn, gate_syn):
		self.v_syn = v_syn
		self.ge_syn = ge_syn
		self.gf_syn = gf_syn
		self.gate_syn = gate_syn

	def print_syn(self):
		np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
		for header in ["v_syn", "ge_syn", "gf_syn", "gate_syn"]:
			print header + ": " + np.array_str(self.__dict__[header])

def benosman_voltage():
	pass

def main():

	# setup
	t = np.arange(0, 1.5, 1e-5) # time (all of this in fixed time frame)

	# generate spike train for RECALL
	s = np.zeros(5) # arbitrary, will fill in

	init_synapses = synapses(s, s, s, s)
	init_synapses.print_syn()

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
