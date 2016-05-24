
import numpy as np

from constants.constants import *
from neural.syn import *

functions = {
    "logarithm" : {
        "t" : 0.5,
        "neuron_names" : ["input", "first", "last", "acc", "output"],
        "synapses" : np.asarray([
            synapse_list("input", "first", np.asarray([
                synapse("V", w_e, T_syn)
            ])),
            synapse_list("first", "first", np.asarray([
                synapse("V", w_i, T_syn)
            ])),
            synapse_list("input", "last", np.asarray([
                synapse("V", 0.5 * w_e, T_syn)
            ])),
            synapse_list("first", "acc", np.asarray([
                synapse("g_e", w_bar_acc, T_syn + T_min)
            ])),
            synapse_list("last", "acc", np.asarray([
                synapse("g_e", -w_bar_acc, T_syn),
                synapse("g_f", g_mult, T_syn),
                synapse("gate", 1, T_syn)
            ])),
            synapse_list("last", "output", np.asarray([
                synapse("V", w_e, 2 * T_syn)
            ])),
            synapse_list("acc", "output", np.asarray([
                synapse("V", w_e, T_syn + T_min)
            ]))
        ]),
        "output_idx" : [4]
    },
    "maximum" : {
        "t" : 1,
        "neuron_names" : ["input", "inputtwo", "larger", "largertwo", "output"],
        "synapses" : np.asarray([
            synapse_list("input", "largertwo", np.asarray([
                synapse("V", 0.5 * w_e, T_syn)
            ])),
            synapse_list("input", "output", np.asarray([
                synapse("V", 0.5 * w_e, T_syn)
            ])),
            synapse_list("inputtwo", "output", np.asarray([
                synapse("V", 0.5 * w_e, T_syn)
            ])),
            synapse_list("inputtwo", "larger", np.asarray([
                synapse("V", 0.5 * w_e, T_syn + T_min)
            ])),
            synapse_list("larger", "largertwo", np.asarray([
                synapse("V", w_i, T_syn),
            ])),
            synapse_list("largertwo", "larger", np.asarray([
                synapse("V", w_i, T_syn)
            ]))
        ]),
        "output_idx" : [4]
    },
    "inverting_memory" : {
        "t" : 0.8,
        "neuron_names" : ["input", "first", "last", "acc",
            "recall", "output"],
        "synapses" : np.asarray([
            synapse_list("input", "first", np.asarray([
                synapse("V", w_e, T_syn)
            ])),
            synapse_list("first", "first", np.asarray([
                synapse("V", w_i, T_syn)
            ])),
            synapse_list("input", "last", np.asarray([
                synapse("V", 0.5 * w_e, T_syn)
            ])),
            synapse_list("first", "acc", np.asarray([
                synapse("g_e", w_acc, T_syn + T_min)
            ])),
            synapse_list("last", "acc", np.asarray([
                synapse("g_e", -w_acc, T_syn),
            ])),
            synapse_list("acc", "output", np.asarray([
                synapse("V", w_e, T_syn)
            ])),
            synapse_list("recall", "acc", np.asarray([
                synapse("g_e", w_acc, T_syn)
            ])),
            synapse_list("recall", "output", np.asarray([
                synapse("V", w_e, 2 * T_syn)
            ]))
        ]),
        "output_idx" : [5]
    },
    "non_inverting_memory" : {
        "t" : 0.8,
        "neuron_names" : ["input", "first", "last", "acc", "acctwo",
        "recall", "ready", "output"],
        "synapses" : np.asarray([
            synapse_list("input", "first", np.asarray([
                synapse("V", w_e, T_syn)
            ])),
            synapse_list("first", "first", np.asarray([
                synapse("V", w_i, T_syn)
            ])),
            synapse_list("input", "last", np.asarray([
                synapse("V", 0.5 * w_e, T_syn)
            ])),
            synapse_list("first", "acc", np.asarray([
                synapse("g_e", w_acc, T_syn)
            ])),
            synapse_list("acc", "acctwo", np.asarray([
                synapse("g_e", -w_acc, T_syn)
            ])),
            synapse_list("last", "acctwo", np.asarray([
                synapse("g_e", w_acc, T_syn)
            ])),
            synapse_list("acc", "ready", np.asarray([
                synapse("V", w_e, T_syn)
            ])),
            synapse_list("recall", "acctwo", np.asarray([
                synapse("g_e", w_acc, T_syn)
            ])),
            synapse_list("recall", "output", np.asarray([
                synapse("V", w_e, T_syn)
            ])),
            synapse_list("acctwo", "output", np.asarray([
                synapse("V", w_e, T_syn)
            ]))
        ]),
        "output_idx" : [7]
    },
    "synchronizer" : {
        "t" : 1,
        "neuron_names" : ["inputone", "inputtwo", "outputone", "outputtwo", "_sync"],
        "synapses" : np.asarray([]),
        "output_idx" : [2, 3],
        "subnets" : [{
            "name" : "non_inverting_memory",
            "synapses" : np.asarray([
                synapse_list("inputone", "input", np.asarray([
                    synapse("V", w_e, T_syn)
                ]), 2),
                synapse_list("output", "outputone", np.asarray([
                    synapse("V", w_e, T_syn)
                ]), 3),
                synapse_list("_sync", "recall", np.asarray([
                    synapse("V", w_e, T_syn)
                ]), 2),
                synapse_list("ready", "_sync", np.asarray([
                    synapse("V", 0.5 * w_e, T_syn)
                ]), 3)
            ])
        }, {
            "name" : "non_inverting_memory",
            "synapses" : np.asarray([
                synapse_list("inputtwo", "input", np.asarray([
                    synapse("V", w_e, T_syn)
                ]), 2),
                synapse_list("output", "outputtwo", np.asarray([
                    synapse("V", w_e, T_syn)
                ]), 3),
                synapse_list("_sync", "recall", np.asarray([
                    synapse("V", w_e, T_syn)
                ]), 2),
                synapse_list("ready", "_sync", np.asarray([
                    synapse("V", 0.5 * w_e, T_syn)
                ]), 3)
            ])
        }]
    },
    "full_subtractor" : {
        "t" : 1,
        "neuron_names" : ["inputone", "inputtwo", "syncone",
            "synctwo", "inbone", "inbtwo", "output+", "output-", "zero"],
        "output_idx" : [6, 7],
        "synapses" : np.asarray([
            synapse_list("inputone", "syncone", np.asarray([
                synapse("V", 0.5 * w_e, T_syn)
            ])),
            synapse_list("inputtwo", "synctwo", np.asarray([
                synapse("V", 0.5 * w_e, T_syn)
            ])),
            synapse_list("syncone", "output+", np.asarray([
                synapse("V", w_e, T_min + 3 * T_syn + 2 * T_neu)
            ])),
            synapse_list("syncone", "inbone", np.asarray([
                synapse("V", w_e, T_syn)
            ])),
            synapse_list("syncone", "output-", np.asarray([
                synapse("V", w_e, 3 * T_syn + 2 * T_neu)
            ])),
            synapse_list("syncone", "inbtwo", np.asarray([
                synapse("V", w_i, T_syn)
            ])),
            synapse_list("synctwo", "inbone", np.asarray([
                synapse("V", w_i, T_syn)
            ])),
            synapse_list("synctwo", "output+", np.asarray([
                synapse("V", w_e, 3 * T_syn + 2 * T_neu)
            ])),
            synapse_list("synctwo", "inbtwo", np.asarray([
                synapse("V", w_e, T_syn)
            ])),
            synapse_list("synctwo", "output-", np.asarray([
                synapse("V", w_e, T_min + 3 * T_syn + 2 * T_neu)
            ])),
            synapse_list("inbone", "output+", np.asarray([
                synapse("V", 2 * w_i, T_syn)
            ])),
            synapse_list("inbtwo", "output-", np.asarray([
                synapse("V", 2 * w_i, T_syn)
            ])),
            synapse_list("output+", "inbtwo", np.asarray([
                synapse("V", 0.5 * w_e, T_syn)
            ])),
            synapse_list("output-", "inbone", np.asarray([
                synapse("V", 0.5 * w_e, T_syn)
            ])),
            synapse_list("zero", "zero", np.asarray([
                synapse("V", w_e, T_neu)
            ])),
            synapse_list("synctwo", "zero", np.asarray([
                synapse("V", 0.5 * w_e, T_neu),
                synapse("V", 0.5 * w_i, 2 * T_neu)
            ])),
            synapse_list("syncone", "zero", np.asarray([
                synapse("V", 0.5 * w_e, T_neu),
                synapse("V", 0.5 * w_i, 2 * T_neu)
            ])),
            synapse_list("zero", "inbtwo", np.asarray([
                synapse("V", w_i, T_neu),
            ])),
            synapse_list("zero", "output-", np.asarray([
                synapse("V", 2 * w_i, T_neu),
            ]))
        ])
    },
    "linear_combination" : {
        "t" : 1,
        "neuron_names" : ["inputone+", "inputone-", "inputtwo+", "inputtwo-",
            "inputone+first", "inputone+last", "inputone-first", "inputone-last",
            "inputtwo+first", "inputtwo+last", "inputtwo-first", "inputtwo-last",
            "sync", "accone+", "accone-", "acctwo+", "acctwo-", "inter+", "inter-",
            "output+", "output-", "start", "bridgeone", "bridgetwo"
        ],
        "output_idx" : [19, 20],
        "synapses" : np.asarray([
            synapse_list("inputone+", "inputone+first", np.asarray([
                synapse("V", w_e, T_syn)
            ])),
            synapse_list("inputone+", "inputone+last", np.asarray([
                synapse("V", 0.5 * w_e, T_syn)
            ])),
            synapse_list("inputone-", "inputone-first", np.asarray([
                synapse("V", w_e, T_syn)
            ])),
            synapse_list("inputone-", "inputone-last", np.asarray([
                synapse("V", 0.5 * w_e, T_syn)
            ])),
            synapse_list("inputtwo+", "inputtwo+first", np.asarray([
                synapse("V", w_e, T_syn)
            ])),
            synapse_list("inputtwo+", "inputtwo+last", np.asarray([
                synapse("V", 0.5 * w_e, T_syn)
            ])),
            synapse_list("inputtwo-", "inputtwo-first", np.asarray([
                synapse("V", w_e, T_syn)
            ])),
            synapse_list("inputtwo-", "inputtwo-last", np.asarray([
                synapse("V", 0.5 * w_e, T_syn)
            ])),
            synapse_list("inputone+last", "sync", np.asarray([
                synapse("V", 0.5 * w_e, T_syn)
            ])),
            synapse_list("inputone-last", "sync", np.asarray([
                synapse("V", 0.5 * w_e, T_syn)
            ])),
            synapse_list("inputtwo+last", "sync", np.asarray([
                synapse("V", 0.5 * w_e, T_syn)
            ])),
            synapse_list("inputtwo-last", "sync", np.asarray([
                synapse("V", 0.5 * w_e, T_syn)
            ])),
            synapse_list("sync", "accone+", np.asarray([
                synapse("g_e", w_acc, T_syn)
            ])),
            synapse_list("sync", "accone-", np.asarray([
                synapse("g_e", w_acc, T_syn)
            ])),
            synapse_list("sync", "acctwo+", np.asarray([
                synapse("g_e", w_acc, T_syn)
            ])),
            synapse_list("sync", "acctwo-", np.asarray([
                synapse("g_e", w_acc, T_syn)
            ])),
            synapse_list("inputone+first", "inputone+first", np.asarray([
                synapse("V", w_i, T_syn)
            ])),
            synapse_list("inputone-first", "inputone-first", np.asarray([
                synapse("V", w_i, T_syn)
            ])),
            synapse_list("inputtwo+first", "inputtwo+first", np.asarray([
                synapse("V", w_i, T_syn)
            ])),
            synapse_list("inputtwo-first", "inputtwo-first", np.asarray([
                synapse("V", w_i, T_syn)
            ])),
            synapse_list("inputone+first", "accone+", np.asarray([
                synapse("g_e", alpha0 * w_acc, T_syn + T_min)
            ])),
            synapse_list("inputone+last", "accone+", np.asarray([
                synapse("g_e", -alpha0 * w_acc, T_syn)
            ])),
            synapse_list("inputone-first", "accone-", np.asarray([
                synapse("g_e", alpha0 * w_acc, T_syn + T_min)
            ])),
            synapse_list("inputone-last", "accone-", np.asarray([
                synapse("g_e", -alpha0 * w_acc, T_syn)
            ])),
            synapse_list("inputtwo+first", "accone-", np.asarray([
                synapse("g_e", alpha0 * w_acc, T_syn + T_min)
            ])),
            synapse_list("inputtwo+last", "accone-", np.asarray([
                synapse("g_e", -alpha0 * w_acc, T_syn)
            ])),
            synapse_list("inputtwo-first", "accone+", np.asarray([
                synapse("g_e", alpha0 * w_acc, T_syn + T_min)
            ])),
            synapse_list("inputtwo-last", "accone+", np.asarray([
                synapse("g_e", -alpha0 * w_acc, T_syn)
            ])),
            synapse_list("accone+", "inter+", np.asarray([
                synapse("V", w_e, T_syn)
            ])),
            synapse_list("accone-", "inter-", np.asarray([
                synapse("V", w_e, T_syn)
            ])),
            synapse_list("acctwo+", "inter+", np.asarray([
                synapse("V", w_e, T_syn + T_min)
            ])),
            synapse_list("acctwo-", "inter-", np.asarray([
                synapse("V", w_e, T_syn + T_min)
            ])),
            synapse_list("start", "start", np.asarray([
                synapse("V", w_i, T_syn)
            ]))
        ]),
        "subnets" : [{
            "name" : "synchronizer",
            "synapses" : np.asarray([
                synapse_list("inter+", "inputone", np.asarray([
                    synapse("V", w_e, T_syn)
                ]), 2),
                synapse_list("inter-", "inputtwo", np.asarray([
                    synapse("V", w_e, T_syn)
                ]), 2),
                synapse_list("outputone", "bridgeone", np.asarray([
                    synapse("V", w_e, 0)
                ]), 3),
                synapse_list("outputtwo", "bridgetwo", np.asarray([
                    synapse("V", w_e, 0)
                ]), 3)
            ])
        }, {
            "name" : "full_subtractor",
            "synapses" : np.asarray([
                synapse_list("bridgeone", "inputone", np.asarray([
                    synapse("V", w_e, 0)
                ]), 2),
                synapse_list("bridgetwo", "inputtwo", np.asarray([
                    synapse("V", w_e, 0)
                ]), 2),
                synapse_list("output+", "output+", np.asarray([
                    synapse("V", w_e, T_syn)
                ]), 3),
                synapse_list("output+", "start", np.asarray([
                    synapse("V", w_e, T_syn)
                ]), 3),
                synapse_list("output-", "output-", np.asarray([
                    synapse("V", w_e, T_syn)
                ]), 3),
                synapse_list("output-", "start", np.asarray([
                    synapse("V", w_e, T_syn)
                ]), 3)
            ])
        }]
    }
}
