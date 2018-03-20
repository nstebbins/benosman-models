import numpy as np

from neural.synapse import *
from neural.constants import *

functions = {
    "logarithm": {
        "t": 0.5,
        "neuron_names": ["input", "first", "last", "acc", "output"],
        "synapses": np.asarray([
            SynapseList("input", "first", np.asarray([
                Synapse("V", w_e, T_syn)
            ])),
            SynapseList("first", "first", np.asarray([
                Synapse("V", w_i, T_syn)
            ])),
            SynapseList("input", "last", np.asarray([
                Synapse("V", 0.5 * w_e, T_syn)
            ])),
            SynapseList("first", "acc", np.asarray([
                Synapse("g_e", w_bar_acc, T_syn + T_min)
            ])),
            SynapseList("last", "acc", np.asarray([
                Synapse("g_e", -w_bar_acc, T_syn),
                Synapse("g_f", g_mult, T_syn),
                Synapse("gate", 1, T_syn)
            ])),
            SynapseList("last", "output", np.asarray([
                Synapse("V", w_e, 2 * T_syn)
            ])),
            SynapseList("acc", "output", np.asarray([
                Synapse("V", w_e, T_syn + T_min)
            ]))
        ]),
        "output_idx": [4]
    },
    "maximum": {
        "t": 1,
        "neuron_names": ["input", "inputtwo", "larger", "largertwo", "output"],
        "synapses": np.asarray([
            SynapseList("input", "largertwo", np.asarray([
                Synapse("V", 0.5 * w_e, T_syn)
            ])),
            SynapseList("input", "output", np.asarray([
                Synapse("V", 0.5 * w_e, T_syn)
            ])),
            SynapseList("inputtwo", "output", np.asarray([
                Synapse("V", 0.5 * w_e, T_syn)
            ])),
            SynapseList("inputtwo", "larger", np.asarray([
                Synapse("V", 0.5 * w_e, T_syn + T_min)
            ])),
            SynapseList("larger", "largertwo", np.asarray([
                Synapse("V", w_i, T_syn),
            ])),
            SynapseList("largertwo", "larger", np.asarray([
                Synapse("V", w_i, T_syn)
            ]))
        ]),
        "output_idx": [4]
    },
    "inverting_memory": {
        "t": 0.8,
        "neuron_names": ["input", "first", "last", "acc",
                         "recall", "output"],
        "synapses": np.asarray([
            SynapseList("input", "first", np.asarray([
                Synapse("V", w_e, T_syn)
            ])),
            SynapseList("first", "first", np.asarray([
                Synapse("V", w_i, T_syn)
            ])),
            SynapseList("input", "last", np.asarray([
                Synapse("V", 0.5 * w_e, T_syn)
            ])),
            SynapseList("first", "acc", np.asarray([
                Synapse("g_e", w_acc, T_syn + T_min)
            ])),
            SynapseList("last", "acc", np.asarray([
                Synapse("g_e", -w_acc, T_syn),
            ])),
            SynapseList("acc", "output", np.asarray([
                Synapse("V", w_e, T_syn)
            ])),
            SynapseList("recall", "acc", np.asarray([
                Synapse("g_e", w_acc, T_syn)
            ])),
            SynapseList("recall", "output", np.asarray([
                Synapse("V", w_e, 2 * T_syn)
            ]))
        ]),
        "output_idx": [5]
    },
    "non_inverting_memory": {
        "t": 0.8,
        "neuron_names": ["input", "first", "last", "acc", "acctwo",
                         "recall", "ready", "output"],
        "synapses": np.asarray([
            SynapseList("input", "first", np.asarray([
                Synapse("V", w_e, T_syn)
            ])),
            SynapseList("first", "first", np.asarray([
                Synapse("V", w_i, T_syn)
            ])),
            SynapseList("input", "last", np.asarray([
                Synapse("V", 0.5 * w_e, T_syn)
            ])),
            SynapseList("first", "acc", np.asarray([
                Synapse("g_e", w_acc, T_syn)
            ])),
            SynapseList("acc", "acctwo", np.asarray([
                Synapse("g_e", -w_acc, T_syn)
            ])),
            SynapseList("last", "acctwo", np.asarray([
                Synapse("g_e", w_acc, T_syn)
            ])),
            SynapseList("acc", "ready", np.asarray([
                Synapse("V", w_e, T_syn)
            ])),
            SynapseList("recall", "acctwo", np.asarray([
                Synapse("g_e", w_acc, T_syn)
            ])),
            SynapseList("recall", "output", np.asarray([
                Synapse("V", w_e, T_syn)
            ])),
            SynapseList("acctwo", "output", np.asarray([
                Synapse("V", w_e, T_syn)
            ]))
        ]),
        "output_idx": [7]
    },
    "full_subtractor": {
        "t": 1,
        "neuron_names": ["inputone", "inputtwo", "syncone",
                         "synctwo", "inbone", "inbtwo", "output+", "output-", "zero"],
        "output_idx": [6, 7],
        "synapses": np.asarray([
            SynapseList("inputone", "syncone", np.asarray([
                Synapse("V", 0.5 * w_e, T_syn)
            ])),
            SynapseList("inputtwo", "synctwo", np.asarray([
                Synapse("V", 0.5 * w_e, T_syn)
            ])),
            SynapseList("syncone", "output+", np.asarray([
                Synapse("V", w_e, T_min + 3 * T_syn + 2 * T_neu)
            ])),
            SynapseList("syncone", "inbone", np.asarray([
                Synapse("V", w_e, T_syn)
            ])),
            SynapseList("syncone", "output-", np.asarray([
                Synapse("V", w_e, 3 * T_syn + 2 * T_neu)
            ])),
            SynapseList("syncone", "inbtwo", np.asarray([
                Synapse("V", w_i, T_syn)
            ])),
            SynapseList("synctwo", "inbone", np.asarray([
                Synapse("V", w_i, T_syn)
            ])),
            SynapseList("synctwo", "output+", np.asarray([
                Synapse("V", w_e, 3 * T_syn + 2 * T_neu)
            ])),
            SynapseList("synctwo", "inbtwo", np.asarray([
                Synapse("V", w_e, T_syn)
            ])),
            SynapseList("synctwo", "output-", np.asarray([
                Synapse("V", w_e, T_min + 3 * T_syn + 2 * T_neu)
            ])),
            SynapseList("inbone", "output+", np.asarray([
                Synapse("V", 2 * w_i, T_syn)
            ])),
            SynapseList("inbtwo", "output-", np.asarray([
                Synapse("V", 2 * w_i, T_syn)
            ])),
            SynapseList("output+", "inbtwo", np.asarray([
                Synapse("V", 0.5 * w_e, T_syn)
            ])),
            SynapseList("output-", "inbone", np.asarray([
                Synapse("V", 0.5 * w_e, T_syn)
            ])),
            SynapseList("zero", "zero", np.asarray([
                Synapse("V", w_e, T_neu)
            ])),
            SynapseList("synctwo", "zero", np.asarray([
                Synapse("V", 0.5 * w_e, T_neu),
                Synapse("V", 0.5 * w_i, 2 * T_neu)
            ])),
            SynapseList("syncone", "zero", np.asarray([
                Synapse("V", 0.5 * w_e, T_neu),
                Synapse("V", 0.5 * w_i, 2 * T_neu)
            ])),
            SynapseList("zero", "inbtwo", np.asarray([
                Synapse("V", w_i, T_neu),
            ])),
            SynapseList("zero", "output-", np.asarray([
                Synapse("V", 2 * w_i, T_neu),
            ]))
        ])
    }
}
