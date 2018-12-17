import numpy as np
import pytest

import neuralkernel.constants as constants
import neuralkernel.neuron as neuron
import neuralkernel.synapse as synapse


class TestSynapseMatrix:

    @pytest.fixture
    def _t(self):
        return np.arange(3)

    @pytest.fixture
    def _synapse_matrix(self, _t):
        synapses = {
            synapse.SynapseGroupKey("one", "two"): [
                synapse.Synapse("v", constants.V_THRESHOLD, 0)
            ]
        }
        neurons = {neuron_name: neuron.Neuron(neuron_name, _t) for
                   neuron_name in ['one', 'two']}
        neurons['one'].set_spikes(0)
        return synapse.SynapseMatrix(neurons, synapses)

    def test_simulate(self, _t, _synapse_matrix):
        # actual
        _synapse_matrix.simulate(_t)
        # expected
        expected_v1 = np.asarray([constants.V_THRESHOLD, 0, 0])
        expected_v2 = np.asarray([0, constants.V_THRESHOLD, 0])
        # test
        np.testing.assert_almost_equal(_synapse_matrix.neurons['one'].v,
                                       expected_v1)
        np.testing.assert_almost_equal(_synapse_matrix.neurons['two'].v,
                                       expected_v2)

    def test_synapse_prop(self, _synapse_matrix):
        # actual
        synapse_time = 1
        _synapse_matrix.propagate_synapse(
            _synapse_matrix.neurons['one'],
            _synapse_matrix.neurons['two'], synapse_time)
        # expected
        expected_v = np.asarray([0, 0, constants.V_THRESHOLD])
        # test
        np.testing.assert_almost_equal(_synapse_matrix.neurons['two'].v,
                                       expected_v)
