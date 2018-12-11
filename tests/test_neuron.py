import unittest

import numpy as np

import neuralkernel.constants as constants
import neuralkernel.neuron as neuron


class TestNeuron(unittest.TestCase):
    def setUp(self):
        self.neuron = neuron.Neuron('one', np.arange(0, 3))

    def test_set_spikes(self):
        # actual
        self.neuron.set_spikes([1])
        # expected
        expected_v = np.asarray([0, constants.V_THRESHOLD, 0])
        # test
        np.testing.assert_almost_equal(self.neuron.v, expected_v)

    def test_simulate(self):
        # actual
        self.neuron.ge[0] = 100
        self.neuron.v[0] = 0.05
        self.neuron.simulate(1)
        # expected
        expected_v = np.asarray([0.05, 0.051, 0])
        # test
        np.testing.assert_almost_equal(self.neuron.v, expected_v)


if __name__ == '__main__':
    unittest.main()
