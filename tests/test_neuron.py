import numpy as np
import pytest

import neuralkernel.constants as constants
import neuralkernel.neuron as neuron


class TestNeuron:

    @pytest.fixture
    def _neuron(self):
        return neuron.Neuron('one', np.arange(0, 3))

    def test_set_spikes(self, _neuron):
        # actual
        _neuron.set_spikes([1])
        # expected
        expected_v = np.asarray([0, constants.V_THRESHOLD, 0])
        # test
        np.testing.assert_almost_equal(_neuron.v, expected_v)

    def test_simulate(self, _neuron):
        # actual
        _neuron.ge[0] = 100
        _neuron.v[0] = 0.05
        _neuron.simulate(1)
        # expected
        expected_v = np.asarray([0.05, 0.051, 0])
        # test
        np.testing.assert_almost_equal(_neuron.v, expected_v)
