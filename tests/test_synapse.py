import unittest

import numpy as np

import neuralkernel.constants as constants
import neuralkernel.neuron as neuron
import neuralkernel.synapse as synapse


class TestSynapseMatrix(unittest.TestCase):
    def setUp(self):
        synapses = {
            synapse.SynapseGroupKey("one", "two"): [
                synapse.Synapse("v", constants.V_THRESHOLD, 0)
            ]
        }
        self.t = np.arange(0, 3)
        neurons = {neuron_name: neuron.Neuron(neuron_name, self.t) for
                   neuron_name in ['one', 'two']}
        neurons['one'].set_spikes(0)
        self.synapse_matrix = synapse.SynapseMatrix(neurons, synapses)

    def test_simulate(self):
        # actual
        self.synapse_matrix.simulate(self.t)
        # expected
        expected_v1 = np.asarray([constants.V_THRESHOLD, 0, 0])
        expected_v2 = np.asarray([0, constants.V_THRESHOLD, 0])
        # test
        np.testing.assert_almost_equal(self.synapse_matrix.neurons['one'].v,
                                       expected_v1)
        np.testing.assert_almost_equal(self.synapse_matrix.neurons['two'].v,
                                       expected_v2)

    def test_synapse_prop(self):
        # actual
        synapse_time = 1
        self.synapse_matrix.propagate_synapse(
            self.synapse_matrix.neurons['one'],
            self.synapse_matrix.neurons['two'], synapse_time)
        # expected
        expected_v = np.asarray([0, 0, constants.V_THRESHOLD])
        # test
        np.testing.assert_almost_equal(self.synapse_matrix.neurons['two'].v,
                                       expected_v)
