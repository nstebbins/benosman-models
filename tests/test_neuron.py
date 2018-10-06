import unittest

import numpy as np

import neuralkernel.constants as constants
import neuralkernel.neuron as neuron


class TestNeuron(unittest.TestCase):
    def setUp(self):
        t = np.arange(0, 3)
        self.neuron = neuron.Neuron('one', t)

    def test_populate_spikes_from_data(self):
        # actual
        spike_indices = [1]
        self.neuron.populate_spikes_from_data(spike_indices)
        # expected
        expected_v = np.asarray([0, constants.V_THRESHOLD, 0])
        # test
        np.testing.assert_almost_equal(self.neuron.v, expected_v)

    def test_next_v(self):
        # actual
        self.neuron.g_e[0] = 100
        self.neuron.v[0] = 0.05
        self.neuron.next_v(1)
        # expected
        expected_v = np.asarray([0.05, 0.051, 0])
        # test
        np.testing.assert_almost_equal(self.neuron.v, expected_v)


if __name__ == '__main__':
    unittest.main()
