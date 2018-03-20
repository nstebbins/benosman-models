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
    "synchronizer": {
        "t": 1,
        "neuron_names": ["inputone", "inputtwo", "outputone", "outputtwo", "_sync"],
        "synapses": np.asarray([]),
        "output_idx": [2, 3],
        "subnets": [{
            "name": "non_inverting_memory",
            "synapses": np.asarray([
                SynapseList("inputone", "input", np.asarray([
                    Synapse("V", w_e, T_syn)
                ]), 2),
                SynapseList("output", "outputone", np.asarray([
                    Synapse("V", w_e, T_syn)
                ]), 3),
                SynapseList("_sync", "recall", np.asarray([
                    Synapse("V", w_e, T_syn)
                ]), 2),
                SynapseList("ready", "_sync", np.asarray([
                    Synapse("V", 0.5 * w_e, T_syn)
                ]), 3)
            ])
        }, {
            "name": "non_inverting_memory",
            "synapses": np.asarray([
                SynapseList("inputtwo", "input", np.asarray([
                    Synapse("V", w_e, T_syn)
                ]), 2),
                SynapseList("output", "outputtwo", np.asarray([
                    Synapse("V", w_e, T_syn)
                ]), 3),
                SynapseList("_sync", "recall", np.asarray([
                    Synapse("V", w_e, T_syn)
                ]), 2),
                SynapseList("ready", "_sync", np.asarray([
                    Synapse("V", 0.5 * w_e, T_syn)
                ]), 3)
            ])
        }]
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
    },
    "integrator": {  # where init = always positive
        "t": 2,
        "neuron_names": ["input+", "input-", "start", "init",
                         "output+", "output-", "new_input", "bridge+", "bridge-"],
        "output_idx": [4, 5],
        "synapses": np.asarray([]),
        "subnets": [{
            "name": "linear_combination",
            "synapses": np.asarray([
                SynapseList("input+", "inputtwo+", np.asarray([
                    Synapse("V", w_e, T_syn)
                ]), 2),
                SynapseList("input-", "inputtwo-", np.asarray([
                    Synapse("V", w_e, T_syn)
                ]), 2),
                SynapseList("start", "inputtwo+", np.asarray([
                    Synapse("V", w_e, T_syn),
                    Synapse("V", w_e, T_syn + T_min)
                ]), 2),
                SynapseList("init", "inputone+", np.asarray([
                    Synapse("V", w_e, T_syn)
                ]), 2),
                SynapseList("output+", "bridge+", np.asarray([
                    Synapse("V", w_e, 0)
                ]), 3),
                SynapseList("output-", "bridge-", np.asarray([
                    Synapse("V", w_e, 0)
                ]), 3),
                SynapseList("bridge+", "inputone+", np.asarray([
                    Synapse("V", w_e, T_syn)
                ]), 2),
                SynapseList("bridge-", "inputone-", np.asarray([
                    Synapse("V", w_e, T_syn)
                ]), 2),
                SynapseList("start", "new_input", np.asarray([
                    Synapse("V", w_e, T_syn)
                ]), 3),
                SynapseList("output+", "output+", np.asarray([
                    Synapse("V", w_e, T_syn)
                ]), 3),
                SynapseList("output-", "output-", np.asarray([
                    Synapse("V", w_e, T_syn)
                ]), 3)
            ])
        }]
    },
    "linear_combination": {
        "t": 1,
        "neuron_names": ["inputone+", "inputone-", "inputtwo+", "inputtwo-",
                         "inputone+first", "inputone+last", "inputone-first", "inputone-last",
                         "inputtwo+first", "inputtwo+last", "inputtwo-first", "inputtwo-last",
                         "sync", "accone+", "accone-", "acctwo+", "acctwo-", "inter+", "inter-",
                         "output+", "output-", "start", "bridgeone", "bridgetwo"
                         ],
        "output_idx": [19, 20],
        "synapses": np.asarray([
            SynapseList("inputone+", "inputone+first", np.asarray([
                Synapse("V", w_e, T_syn)
            ])),
            SynapseList("inputone+", "inputone+last", np.asarray([
                Synapse("V", 0.5 * w_e, T_syn)
            ])),
            SynapseList("inputone-", "inputone-first", np.asarray([
                Synapse("V", w_e, T_syn)
            ])),
            SynapseList("inputone-", "inputone-last", np.asarray([
                Synapse("V", 0.5 * w_e, T_syn)
            ])),
            SynapseList("inputtwo+", "inputtwo+first", np.asarray([
                Synapse("V", w_e, T_syn)
            ])),
            SynapseList("inputtwo+", "inputtwo+last", np.asarray([
                Synapse("V", 0.5 * w_e, T_syn)
            ])),
            SynapseList("inputtwo-", "inputtwo-first", np.asarray([
                Synapse("V", w_e, T_syn)
            ])),
            SynapseList("inputtwo-", "inputtwo-last", np.asarray([
                Synapse("V", 0.5 * w_e, T_syn)
            ])),
            SynapseList("inputone+last", "sync", np.asarray([
                Synapse("V", 0.5 * w_e, T_syn)
            ])),
            SynapseList("inputone-last", "sync", np.asarray([
                Synapse("V", 0.5 * w_e, T_syn)
            ])),
            SynapseList("inputtwo+last", "sync", np.asarray([
                Synapse("V", 0.5 * w_e, T_syn)
            ])),
            SynapseList("inputtwo-last", "sync", np.asarray([
                Synapse("V", 0.5 * w_e, T_syn)
            ])),
            SynapseList("sync", "accone+", np.asarray([
                Synapse("g_e", w_acc, T_syn)
            ])),
            SynapseList("sync", "accone-", np.asarray([
                Synapse("g_e", w_acc, T_syn)
            ])),
            SynapseList("sync", "acctwo+", np.asarray([
                Synapse("g_e", w_acc, T_syn)
            ])),
            SynapseList("sync", "acctwo-", np.asarray([
                Synapse("g_e", w_acc, T_syn)
            ])),
            SynapseList("inputone+first", "inputone+first", np.asarray([
                Synapse("V", w_i, T_syn)
            ])),
            SynapseList("inputone-first", "inputone-first", np.asarray([
                Synapse("V", w_i, T_syn)
            ])),
            SynapseList("inputtwo+first", "inputtwo+first", np.asarray([
                Synapse("V", w_i, T_syn)
            ])),
            SynapseList("inputtwo-first", "inputtwo-first", np.asarray([
                Synapse("V", w_i, T_syn)
            ])),
            SynapseList("inputone+first", "accone+", np.asarray([
                Synapse("g_e", alpha0 * w_acc, T_syn + T_min)
            ])),
            SynapseList("inputone+last", "accone+", np.asarray([
                Synapse("g_e", -alpha0 * w_acc, T_syn)
            ])),
            SynapseList("inputone-first", "accone-", np.asarray([
                Synapse("g_e", alpha0 * w_acc, T_syn + T_min)
            ])),
            SynapseList("inputone-last", "accone-", np.asarray([
                Synapse("g_e", -alpha0 * w_acc, T_syn)
            ])),
            SynapseList("inputtwo+first", "accone-", np.asarray([
                Synapse("g_e", alpha0 * w_acc, T_syn + T_min)
            ])),
            SynapseList("inputtwo+last", "accone-", np.asarray([
                Synapse("g_e", -alpha0 * w_acc, T_syn)
            ])),
            SynapseList("inputtwo-first", "accone+", np.asarray([
                Synapse("g_e", alpha0 * w_acc, T_syn + T_min)
            ])),
            SynapseList("inputtwo-last", "accone+", np.asarray([
                Synapse("g_e", -alpha0 * w_acc, T_syn)
            ])),
            SynapseList("accone+", "inter+", np.asarray([
                Synapse("V", w_e, T_syn)
            ])),
            SynapseList("accone-", "inter-", np.asarray([
                Synapse("V", w_e, T_syn)
            ])),
            SynapseList("acctwo+", "inter+", np.asarray([
                Synapse("V", w_e, T_syn + T_min)
            ])),
            SynapseList("acctwo-", "inter-", np.asarray([
                Synapse("V", w_e, T_syn + T_min)
            ])),
            SynapseList("start", "start", np.asarray([
                Synapse("V", w_i, T_syn)
            ]))
        ]),
        "subnets": [{
            "name": "synchronizer",
            "synapses": np.asarray([
                SynapseList("inter+", "inputone", np.asarray([
                    Synapse("V", w_e, T_syn)
                ]), 2),
                SynapseList("inter-", "inputtwo", np.asarray([
                    Synapse("V", w_e, T_syn)
                ]), 2),
                SynapseList("outputone", "bridgeone", np.asarray([
                    Synapse("V", w_e, 0)
                ]), 3),
                SynapseList("outputtwo", "bridgetwo", np.asarray([
                    Synapse("V", w_e, 0)
                ]), 3)
            ])
        }, {
            "name": "full_subtractor",
            "synapses": np.asarray([
                SynapseList("bridgeone", "inputone", np.asarray([
                    Synapse("V", w_e, 0)
                ]), 2),
                SynapseList("bridgetwo", "inputtwo", np.asarray([
                    Synapse("V", w_e, 0)
                ]), 2),
                SynapseList("output+", "output+", np.asarray([
                    Synapse("V", w_e, T_syn)
                ]), 3),
                SynapseList("output+", "start", np.asarray([
                    Synapse("V", w_e, T_syn)
                ]), 3),
                SynapseList("output-", "output-", np.asarray([
                    Synapse("V", w_e, T_syn)
                ]), 3),
                SynapseList("output-", "start", np.asarray([
                    Synapse("V", w_e, T_syn)
                ]), 3)
            ])
        }]
    }
}
