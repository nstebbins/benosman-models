
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
        "neuron_names" : ["input", "input2", "larger1", "larger2", "output"],
        "synapses" : np.asarray([
            synapse_list("input", "larger2", np.asarray([
                synapse("V", 0.5 * w_e, T_syn)
            ])),
            synapse_list("input", "output", np.asarray([
                synapse("V", 0.5 * w_e, T_syn)
            ])),
            synapse_list("input2", "output", np.asarray([
                synapse("V", 0.5 * w_e, T_syn)
            ])),
            synapse_list("input2", "larger1", np.asarray([
                synapse("V", 0.5 * w_e, T_syn + T_min)
            ])),
            synapse_list("larger1", "larger2", np.asarray([
                synapse("V", w_i, T_syn),
            ])),
            synapse_list("larger2", "larger1", np.asarray([
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
        "neuron_names" : ["input", "first", "last", "acc", "acc2",
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
            synapse_list("acc", "acc2", np.asarray([
                synapse("g_e", -w_acc, T_syn)
            ])),
            synapse_list("last", "acc2", np.asarray([
                synapse("g_e", w_acc, T_syn)
            ])),
            synapse_list("acc", "ready", np.asarray([
                synapse("V", w_e, T_syn)
            ])),
            synapse_list("recall", "acc2", np.asarray([
                synapse("g_e", w_acc, T_syn)
            ])),
            synapse_list("recall", "output", np.asarray([
                synapse("V", w_e, T_syn)
            ])),
            synapse_list("acc2", "output", np.asarray([
                synapse("V", w_e, T_syn)
            ]))
        ]),
        "output_idx" : [7]
    },
    "synchronizer" : {
        "t" : 1,
        "neuron_names" : ["input0", "input1", "output0", "output1", "_sync"],
        "synapses" : np.asarray([]),
        "output_idx" : [2, 3],
        "subnets" : [{
            "name" : "non_inverting_memory",
            "n" : 2,
            "synapses" : np.asarray([
                synapse_list("input", "input", np.asarray([
                    synapse("V", w_e, T_syn)
                ]), 2),
                synapse_list("output", "output", np.asarray([
                    synapse("V", w_e, T_syn)
                ]), 4),
                synapse_list("_sync", "recall", np.asarray([
                    synapse("V", w_e, T_syn)
                ]), 2),
                synapse_list("ready", "_sync", np.asarray([
                    synapse("V", 0.5 * w_e, T_syn)
                ]), 4)
            ])
        }]
    },
    "full_subtractor" : {
        "t" : 1,
        "neuron_names" : ["input1", "input2", "zero", "sync1",
            "sync2", "inb1", "inb2", "output_plus", "output_minus"],
        "output_idx" : [0, 1, 7, 8],
        "synapses" : np.asarray([
            synapse_list("input1", "sync1", np.asarray([
                synapse("V", 0.5 * w_e, T_syn)
            ])),
            synapse_list("input2", "sync2", np.asarray([
                synapse("V", 0.5 * w_e, T_syn)
            ])),
            synapse_list("sync1", "zero", np.asarray([
                synapse("V", w_e, T_neu),
                synapse("V", w_i, 2 * T_neu)
            ])),
            synapse_list("sync1", "output_plus", np.asarray([
                synapse("V", w_e, T_min + 3 * T_syn + 2 * T_neu)
            ])),
            synapse_list("sync1", "inb1", np.asarray([
                synapse("V", w_e, T_syn)
            ])),
            synapse_list("sync1", "output_minus", np.asarray([
                synapse("V", w_e, 3 * T_syn + 2 * T_neu)
            ])),
            synapse_list("sync1", "inb2", np.asarray([
                synapse("V", w_i, T_syn)
            ])),
            synapse_list("sync2", "zero", np.asarray([
                synapse("V", w_e, T_neu),
                synapse("V", w_i, 2 * T_neu)
            ])),
            synapse_list("sync2", "inb1", np.asarray([
                synapse("V", w_i, T_syn)
            ])),
            synapse_list("sync2", "output_plus", np.asarray([
                synapse("V", w_e, 3 * T_syn + 2 * T_neu)
            ])),
            synapse_list("sync2", "inb2", np.asarray([
                synapse("V", w_e, T_syn)
            ])),
            synapse_list("sync2", "output_minus", np.asarray([
                synapse("V", w_e, T_min + 3 * T_syn + 2 * T_neu)
            ])),
            synapse_list("zero", "zero", np.asarray([
                synapse("V", w_e, T_neu)
            ])),
            synapse_list("zero", "inb2", np.asarray([
                synapse("V", w_i, T_neu)
            ])),
            synapse_list("zero", "output_minus", np.asarray([
                synapse("V", 2 * w_i, T_neu)
            ])),
            synapse_list("inb1", "output_plus", np.asarray([
                synapse("V", 2 * w_i, T_syn)
            ])),
            synapse_list("inb2", "output_minus", np.asarray([
                synapse("V", 2 * w_i, T_syn)
            ])),
            synapse_list("output_plus", "inb2", np.asarray([
                synapse("V", 0.5 * w_e, T_syn)
            ])),
            synapse_list("output_minus", "inb1", np.asarray([
                synapse("V", 0.5 * w_e, T_syn)
            ]))
        ])
    },
    "linear_combination" : {
        "t" : 1,
        "neuron_names" : ["input0+", "input0-", "input1+", "input1-",
            "input0+first", "input0+last", "input0-first", "input0-last",
            "input1+first", "input1+last", "input1-first", "input1-last",
            "sync", "acc1+", "acc1-", "acc2+", "acc2-", "inter+", "inter-",
            "output+", "output-"
        ],
        "output_idx" : [19, 20],
        "synapses" : np.asarray([
            synapse_list("input0+", "input0+first", np.asarray([
                synapse("V", w_e, T_syn)
            ])),
            synapse_list("input0+", "input0+last", np.asarray([
                synapse("V", 0.5 * w_e, T_syn)
            ])),
            synapse_list("input0-", "input0-first", np.asarray([
                synapse("V", w_e, T_syn)
            ])),
            synapse_list("input0-", "input0-last", np.asarray([
                synapse("V", 0.5 * w_e, T_syn)
            ])),
            synapse_list("input1+", "input1+first", np.asarray([
                synapse("V", w_e, T_syn)
            ])),
            synapse_list("input1+", "input1+last", np.asarray([
                synapse("V", 0.5 * w_e, T_syn)
            ])),
            synapse_list("input1-", "input1-first", np.asarray([
                synapse("V", w_e, T_syn)
            ])),
            synapse_list("input1-", "input1-last", np.asarray([
                synapse("V", 0.5 * w_e, T_syn)
            ])),
            synapse_list("input0+last", "sync", np.asarray([
                synapse("V", 0.5 * w_e, T_syn)
            ])),
            synapse_list("input0-last", "sync", np.asarray([
                synapse("V", 0.5 * w_e, T_syn)
            ])),
            synapse_list("input1+last", "sync", np.asarray([
                synapse("V", 0.5 * w_e, T_syn)
            ])),
            synapse_list("input1-last", "sync", np.asarray([
                synapse("V", 0.5 * w_e, T_syn)
            ])),
            synapse_list("sync", "acc1+", np.asarray([
                synapse("g_e", w_acc, T_syn)
            ])),
            synapse_list("sync", "acc1-", np.asarray([
                synapse("g_e", w_acc, T_syn)
            ])),
            synapse_list("sync", "acc2+", np.asarray([
                synapse("g_e", w_acc, T_syn)
            ])),
            synapse_list("sync", "acc2-", np.asarray([
                synapse("g_e", w_acc, T_syn)
            ])),
            synapse_list("input0+first", "input0+first", np.asarray([
                synapse("V", w_i, T_syn)
            ])),
            synapse_list("input0-first", "input0-first", np.asarray([
                synapse("V", w_i, T_syn)
            ])),
            synapse_list("input1+first", "input1+first", np.asarray([
                synapse("V", w_i, T_syn)
            ])),
            synapse_list("input1-first", "input1-first", np.asarray([
                synapse("V", w_i, T_syn)
            ])),
            synapse_list("input0+first", "acc1+", np.asarray([
                synapse("g_e", alpha0 * w_acc, T_syn + T_min)
            ])),
            synapse_list("input0+last", "acc1+", np.asarray([
                synapse("g_e", -alpha0 * w_acc, T_syn)
            ])),
            synapse_list("input0-first", "acc1-", np.asarray([
                synapse("g_e", alpha0 * w_acc, T_syn + T_min)
            ])),
            synapse_list("input0-last", "acc1-", np.asarray([
                synapse("g_e", -alpha0 * w_acc, T_syn)
            ])),
            synapse_list("input1+first", "acc1-", np.asarray([
                synapse("g_e", alpha0 * w_acc, T_syn + T_min)
            ])),
            synapse_list("input1+last", "acc1-", np.asarray([
                synapse("g_e", -alpha0 * w_acc, T_syn)
            ])),
            synapse_list("input1-first", "acc1+", np.asarray([
                synapse("g_e", alpha0 * w_acc, T_syn + T_min)
            ])),
            synapse_list("input1-last", "acc1+", np.asarray([
                synapse("g_e", -alpha0 * w_acc, T_syn)
            ])),
            synapse_list("acc1+", "inter+", np.asarray([
                synapse("V", w_e, T_syn)
            ])),
            synapse_list("acc1-", "inter-", np.asarray([
                synapse("V", w_e, T_syn)
            ])),
            synapse_list("acc2+", "inter+", np.asarray([
                synapse("V", w_e, T_syn + T_min)
            ])),
            synapse_list("acc2-", "inter-", np.asarray([
                synapse("V", w_e, T_syn + T_min)
            ])),
            synapse_list("start", "start", np.asarray([
                synapse("V", w_i, T_syn)
            ]))
        ]),
        "subnets" : [{
            "name" : "synchronizer",
            "n" : 2,
            "synapses" : np.asarray([
                synapse_list("_inter+", "", np.asarray([

                ]), 2)
            ])
        }, {
            "name" : "synchronizer",
            "n" : 2,
            "synapses" : np.asarray([
                synapse_list("_inter-", "", np.asarray([

                ]), 2)
            ])
        }, {
            "name" : "full_subtractor",
            "n" : 1,
            "synapses" : np.asarray([
                synapse_list("", "", np.asarray([

                ]), 4),
                synapse_list("", "", np.asarray([

                ]), 4),
                synapse_list("", "", np.asarray([

                ]), 4),
                synapse_list("", "", np.asarray([

                ]), 4)
            ])
        }] # NOTE: also need to account for connections between subnets!
    }
}
