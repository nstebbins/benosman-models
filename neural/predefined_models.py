
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
            "synctwo", "inbone", "inbtwo", "output_plus", "output_minus", "zero"],
        "output_idx" : [0, 1, 2, 3, 4, 5, 6, 7, 8],
        "synapses" : np.asarray([
            synapse_list("inputone", "syncone", np.asarray([
                synapse("V", 0.5 * w_e, T_syn)
            ])),
            synapse_list("inputtwo", "synctwo", np.asarray([
                synapse("V", 0.5 * w_e, T_syn)
            ])),
            synapse_list("syncone", "output_plus", np.asarray([
                synapse("V", w_e, T_min + 3 * T_syn + 2 * T_neu)
            ])),
            synapse_list("syncone", "inbone", np.asarray([
                synapse("V", w_e, T_syn)
            ])),
            synapse_list("syncone", "output_minus", np.asarray([
                synapse("V", w_e, 3 * T_syn + 2 * T_neu)
            ])),
            synapse_list("syncone", "inbtwo", np.asarray([
                synapse("V", w_i, T_syn)
            ])),
            synapse_list("synctwo", "inbone", np.asarray([
                synapse("V", w_i, T_syn)
            ])),
            synapse_list("synctwo", "output_plus", np.asarray([
                synapse("V", w_e, 3 * T_syn + 2 * T_neu)
            ])),
            synapse_list("synctwo", "inbtwo", np.asarray([
                synapse("V", w_e, T_syn)
            ])),
            synapse_list("synctwo", "output_minus", np.asarray([
                synapse("V", w_e, T_min + 3 * T_syn + 2 * T_neu)
            ])),
            synapse_list("inbone", "output_plus", np.asarray([
                synapse("V", 2 * w_i, T_syn)
            ])),
            synapse_list("inbtwo", "output_minus", np.asarray([
                synapse("V", 2 * w_i, T_syn)
            ])),
            synapse_list("output_plus", "inbtwo", np.asarray([
                synapse("V", 0.5 * w_e, T_syn)
            ])),
            synapse_list("output_minus", "inbone", np.asarray([
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
            synapse_list("zero", "output_minus", np.asarray([
                synapse("V", 2 * w_i, T_neu),
            ]))
        ])
    },
    "linear_combination" : {
        "t" : 1,
        "neuron_names" : ["input+", "input-", "inputtwo+", "inputtwo-",
            "input+first", "input+last", "input-first", "input-last",
            "inputtwo+first", "inputtwo+last", "inputtwo-first", "inputtwo-last",
            "sync", "accone+", "accone-", "acctwo+", "acctwo-", "inter+", "inter-",
            "output+", "output-", "start"
        ],
        "output_idx" : [12, 13, 17, 18, 22, 23, 24, 25],
        "synapses" : np.asarray([
            synapse_list("input+", "input+first", np.asarray([
                synapse("V", w_e, T_syn)
            ])),
            synapse_list("input+", "input+last", np.asarray([
                synapse("V", 0.5 * w_e, T_syn)
            ])),
            synapse_list("input-", "input-first", np.asarray([
                synapse("V", w_e, T_syn)
            ])),
            synapse_list("input-", "input-last", np.asarray([
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
            synapse_list("input+last", "sync", np.asarray([
                synapse("V", 0.5 * w_e, T_syn)
            ])),
            synapse_list("input-last", "sync", np.asarray([
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
            synapse_list("input+first", "input+first", np.asarray([
                synapse("V", w_i, T_syn)
            ])),
            synapse_list("input-first", "input-first", np.asarray([
                synapse("V", w_i, T_syn)
            ])),
            synapse_list("inputtwo+first", "inputtwo+first", np.asarray([
                synapse("V", w_i, T_syn)
            ])),
            synapse_list("inputtwo-first", "inputtwo-first", np.asarray([
                synapse("V", w_i, T_syn)
            ])),
            synapse_list("input+first", "accone+", np.asarray([
                synapse("g_e", alpha0 * w_acc, T_syn + T_min)
            ])),
            synapse_list("input+last", "accone+", np.asarray([
                synapse("g_e", -alpha0 * w_acc, T_syn)
            ])),
            synapse_list("input-first", "accone-", np.asarray([
                synapse("g_e", alpha0 * w_acc, T_syn + T_min)
            ])),
            synapse_list("input-last", "accone-", np.asarray([
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
                synapse_list("inter+", "input", np.asarray([
                    synapse("V", w_e, T_syn)
                ]), 2),
                synapse_list("inter-", "inputtwo", np.asarray([
                    synapse("V", w_e, T_syn)
                ]), 2)
            ])
        }, {
            "name" : "full_subtractor",
            "neuron_names" : ["inputsubone", "inputsubtwo", "zero", "syncone",
                "synctwo", "inbone", "inbtwo", "outputsub+", "outputsub-"],
            "synapses" : np.asarray([
                synapse_list("output", "inputsubone", np.asarray([
                    synapse("V", w_e, T_syn)
                ]), 2),
                synapse_list("outputtwo", "inputsubtwo", np.asarray([
                    synapse("V", w_e, T_syn)
                ]), 2),
                synapse_list("outputsub+", "output+", np.asarray([
                    synapse("V", w_e, T_syn)
                ]), 4),
                synapse_list("outputsub+", "start", np.asarray([
                    synapse("V", w_e, T_syn)
                ]), 4),
                synapse_list("outputsub-", "output-", np.asarray([
                    synapse("V", w_e, T_syn)
                ]), 4),
                synapse_list("outputsub-", "start", np.asarray([
                    synapse("V", w_e, T_syn)
                ]), 4)
            ])
        }]
    }
}
