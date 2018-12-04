from collections import namedtuple

from .constants import W_E, W_ACC, W_BAR_ACC, W_I, G_MULT, T_SYN, T_MIN, T_NEU
from .synapse import Synapse, SynapseGroup, SynapseMatrixKey

# TODO: change output_idx to output_neuron_name
Network = namedtuple("Network",
                     ("window", "output_idx", "neuron_names", "synapses"))

logarithm = Network(0.5, {"output"},
                    ["input", "first", "last", "acc", "output"],
                    {
                        SynapseMatrixKey("input", "first"): [
                            Synapse("v", W_E, T_SYN)
                        ],
                        SynapseMatrixKey("first", "first"): [
                            Synapse("v", W_I, T_SYN)
                        ],
                        SynapseMatrixKey("input", "last"): [
                            Synapse("v", 0.5 * W_E, T_SYN)
                        ],
                        SynapseMatrixKey("first", "acc"): [
                            Synapse("ge", W_BAR_ACC, T_SYN + T_MIN)
                        ],
                        SynapseMatrixKey("last", "acc"): [
                            Synapse("ge", -W_BAR_ACC, T_SYN),
                            Synapse("gf", G_MULT, T_SYN),
                            Synapse("gate", 1, T_SYN)
                        ],
                        SynapseMatrixKey("last", "output"): [
                            Synapse("v", W_E, 2 * T_SYN)
                        ],
                        SynapseMatrixKey("acc", "output"): [
                            Synapse("v", W_E, T_SYN + T_MIN)
                        ]
                    })

# TODO: update the rest of these guys

maximum = Network(1, {"output"},
                  ["input", "inputtwo", "larger", "largertwo", "output"],
                  [
                      SynapseGroup("input", "largertwo", [
                          Synapse("v", 0.5 * W_E, T_SYN)
                      ]),
                      SynapseGroup("input", "output", [
                          Synapse("v", 0.5 * W_E, T_SYN)
                      ]),
                      SynapseGroup("inputtwo", "output", [
                          Synapse("v", 0.5 * W_E, T_SYN)
                      ]),
                      SynapseGroup("inputtwo", "larger", [
                          Synapse("v", 0.5 * W_E, T_SYN + T_MIN)
                      ]),
                      SynapseGroup("larger", "largertwo", [
                          Synapse("v", W_I, T_SYN),
                      ]),
                      SynapseGroup("largertwo", "larger", [
                          Synapse("v", W_I, T_SYN)
                      ])
                  ])

inverting_mem = Network(0.8, {"output"},
                        ["input", "first", "last", "acc", "recall", "output"],
                        [
                            SynapseGroup("input", "first", [
                                Synapse("v", W_E, T_SYN)
                            ]),
                            SynapseGroup("first", "first", [
                                Synapse("v", W_I, T_SYN)
                            ]),
                            SynapseGroup("input", "last", [
                                Synapse("v", 0.5 * W_E, T_SYN)
                            ]),
                            SynapseGroup("first", "acc", [
                                Synapse("ge", W_ACC, T_SYN + T_MIN)
                            ]),
                            SynapseGroup("last", "acc", [
                                Synapse("ge", -W_ACC, T_SYN),
                            ]),
                            SynapseGroup("acc", "output", [
                                Synapse("v", W_E, T_SYN)
                            ]),
                            SynapseGroup("recall", "acc", [
                                Synapse("ge", W_ACC, T_SYN)
                            ]),
                            SynapseGroup("recall", "output", [
                                Synapse("v", W_E, 2 * T_SYN)
                            ])
                        ])

non_inverting_mem = Network(0.8, {"output"},
                            ["input", "first", "last", "acc", "acctwo",
                             "recall", "ready", "output"],
                            [
                                SynapseGroup("input", "first", [
                                    Synapse("v", W_E, T_SYN)
                                ]),
                                SynapseGroup("first", "first", [
                                    Synapse("v", W_I, T_SYN)
                                ]),
                                SynapseGroup("input", "last", [
                                    Synapse("v", 0.5 * W_E, T_SYN)
                                ]),
                                SynapseGroup("first", "acc", [
                                    Synapse("ge", W_ACC, T_SYN)
                                ]),
                                SynapseGroup("acc", "acctwo", [
                                    Synapse("ge", -W_ACC, T_SYN)
                                ]),
                                SynapseGroup("last", "acctwo", [
                                    Synapse("ge", W_ACC, T_SYN)
                                ]),
                                SynapseGroup("acc", "ready", [
                                    Synapse("v", W_E, T_SYN)
                                ]),
                                SynapseGroup("recall", "acctwo", [
                                    Synapse("ge", W_ACC, T_SYN)
                                ]),
                                SynapseGroup("recall", "output", [
                                    Synapse("v", W_E, T_SYN)
                                ]),
                                SynapseGroup("acctwo", "output", [
                                    Synapse("v", W_E, T_SYN)
                                ])
                            ])

full_subtractor = Network(1, {"output+", "output-"},
                          ["inputone", "inputtwo", "syncone",
                           "synctwo", "inbone", "inbtwo", "output+",
                           "output-", "zero"],
                          [
                              SynapseGroup("inputone", "syncone", [
                                  Synapse("v", 0.5 * W_E, T_SYN)
                              ]),
                              SynapseGroup("inputtwo", "synctwo", [
                                  Synapse("v", 0.5 * W_E, T_SYN)
                              ]),
                              SynapseGroup("syncone", "output+", [
                                  Synapse("v", W_E,
                                          T_MIN + 3 * T_SYN + 2 * T_NEU)
                              ]),
                              SynapseGroup("syncone", "inbone", [
                                  Synapse("v", W_E, T_SYN)
                              ]),
                              SynapseGroup("syncone", "output-", [
                                  Synapse("v", W_E, 3 * T_SYN + 2 * T_NEU)
                              ]),
                              SynapseGroup("syncone", "inbtwo", [
                                  Synapse("v", W_I, T_SYN)
                              ]),
                              SynapseGroup("synctwo", "inbone", [
                                  Synapse("v", W_I, T_SYN)
                              ]),
                              SynapseGroup("synctwo", "output+", [
                                  Synapse("v", W_E, 3 * T_SYN + 2 * T_NEU)
                              ]),
                              SynapseGroup("synctwo", "inbtwo", [
                                  Synapse("v", W_E, T_SYN)
                              ]),
                              SynapseGroup("synctwo", "output-", [
                                  Synapse("v", W_E,
                                          T_MIN + 3 * T_SYN + 2 * T_NEU)
                              ]),
                              SynapseGroup("inbone", "output+", [
                                  Synapse("v", 2 * W_I, T_SYN)
                              ]),
                              SynapseGroup("inbtwo", "output-", [
                                  Synapse("v", 2 * W_I, T_SYN)
                              ]),
                              SynapseGroup("output+", "inbtwo", [
                                  Synapse("v", 0.5 * W_E, T_SYN)
                              ]),
                              SynapseGroup("output-", "inbone", [
                                  Synapse("v", 0.5 * W_E, T_SYN)
                              ]),
                              SynapseGroup("zero", "zero", [
                                  Synapse("v", W_E, T_NEU)
                              ]),
                              SynapseGroup("synctwo", "zero", [
                                  Synapse("v", 0.5 * W_E, T_NEU),
                                  Synapse("v", 0.5 * W_I, 2 * T_NEU)
                              ]),
                              SynapseGroup("syncone", "zero", [
                                  Synapse("v", 0.5 * W_E, T_NEU),
                                  Synapse("v", 0.5 * W_I, 2 * T_NEU)
                              ]),
                              SynapseGroup("zero", "inbtwo", [
                                  Synapse("v", W_I, T_NEU),
                              ]),
                              SynapseGroup("zero", "output-", [
                                  Synapse("v", 2 * W_I, T_NEU),
                              ])
                          ])
