import unittest

import numpy as np

import neuralkernel.constants as constants
import neuralkernel.neuron as neuron
import neuralkernel.synapse as adjmatrix
import neuralkernel.synapse as synapse


class TestAdjMatrix(unittest.TestCase):
    def setUp(self):
        t = np.arange(0, 3)
        synapses = np.asarray([
            synapse.SynapseGroup('one', 'two', np.asarray([
                synapse.Synapse('V', constants.V_THRESHOLD, 0)
            ]))
        ])
        neurons = [neuron.Neuron(neuron_name, t) for neuron_name in
                   ['one', 'two']]
        self.adj_matrix = adjmatrix.SynapseMatrix(neurons, synapses)

    def test_simulate(self):
        # actual
        self.adj_matrix.neurons[0].v[0] = constants.V_THRESHOLD
        self.adj_matrix.simulate()
        # expected
        expected_v1 = np.asarray([constants.V_THRESHOLD, 0, 0])
        expected_v2 = np.asarray([0, constants.V_THRESHOLD, 0])
        # test
        np.testing.assert_almost_equal(self.adj_matrix.neurons[0].v,
                                       expected_v1)
        np.testing.assert_almost_equal(self.adj_matrix.neurons[1].v,
                                       expected_v2)

    def test_synapse_prop(self):
        # actual
        n_from = 0
        n_to = 1
        tj = 0
        syn = self.adj_matrix.synapse_matrix[n_from][n_to].synapses[0]
        self.adj_matrix.synapse_prop(syn, n_to, tj)
        # expected
        expected_v = np.asarray([0, 10, 0])
        # test
        np.testing.assert_almost_equal(self.adj_matrix.neurons[n_to].v,
                                       expected_v)
