from collections import namedtuple

from .constants import W_E, W_ACC, W_BAR_ACC, W_I, G_MULT, T_SYN, T_MIN, T_NEU
from .synapse import Synapse, SynapseList

# TODO: change output_idx to output_neuron_name
Network = namedtuple("Network",
                     ("window", "output_idx", "neuron_names", "synapses"))

logarithm = Network(0.5, {"output"},
                    ["input", "first", "last", "acc", "output"],
                    [
                        SynapseList("input", "first", [
                            Synapse("v", W_E, T_SYN)
                        ]),
                        SynapseList("first", "first", [
                            Synapse("v", W_I, T_SYN)
                        ]),
                        SynapseList("input", "last", [
                            Synapse("v", 0.5 * W_E, T_SYN)
                        ]),
                        SynapseList("first", "acc", [
                            Synapse("ge", W_BAR_ACC, T_SYN + T_MIN)
                        ]),
                        SynapseList("last", "acc", [
                            Synapse("ge", -W_BAR_ACC, T_SYN),
                            Synapse("gf", G_MULT, T_SYN),
                            Synapse("gate", 1, T_SYN)
                        ]),
                        SynapseList("last", "output", [
                            Synapse("v", W_E, 2 * T_SYN)
                        ]),
                        SynapseList("acc", "output", [
                            Synapse("v", W_E, T_SYN + T_MIN)
                        ])
                    ])

maximum = Network(1, {"output"},
                  ["input", "inputtwo", "larger", "largertwo", "output"],
                  [
                      SynapseList("input", "largertwo", [
                          Synapse("v", 0.5 * W_E, T_SYN)
                      ]),
                      SynapseList("input", "output", [
                          Synapse("v", 0.5 * W_E, T_SYN)
                      ]),
                      SynapseList("inputtwo", "output", [
                          Synapse("v", 0.5 * W_E, T_SYN)
                      ]),
                      SynapseList("inputtwo", "larger", [
                          Synapse("v", 0.5 * W_E, T_SYN + T_MIN)
                      ]),
                      SynapseList("larger", "largertwo", [
                          Synapse("v", W_I, T_SYN),
                      ]),
                      SynapseList("largertwo", "larger", [
                          Synapse("v", W_I, T_SYN)
                      ])
                  ])

inverting_mem = Network(0.8, {"output"},
                        ["input", "first", "last", "acc", "recall", "output"],
                        [
                            SynapseList("input", "first", [
                                Synapse("v", W_E, T_SYN)
                            ]),
                            SynapseList("first", "first", [
                                Synapse("v", W_I, T_SYN)
                            ]),
                            SynapseList("input", "last", [
                                Synapse("v", 0.5 * W_E, T_SYN)
                            ]),
                            SynapseList("first", "acc", [
                                Synapse("ge", W_ACC, T_SYN + T_MIN)
                            ]),
                            SynapseList("last", "acc", [
                                Synapse("ge", -W_ACC, T_SYN),
                            ]),
                            SynapseList("acc", "output", [
                                Synapse("v", W_E, T_SYN)
                            ]),
                            SynapseList("recall", "acc", [
                                Synapse("ge", W_ACC, T_SYN)
                            ]),
                            SynapseList("recall", "output", [
                                Synapse("v", W_E, 2 * T_SYN)
                            ])
                        ])

non_inverting_mem = Network(0.8, {"output"},
                            ["input", "first", "last", "acc", "acctwo",
                             "recall", "ready", "output"],
                            [
                                SynapseList("input", "first", [
                                    Synapse("v", W_E, T_SYN)
                                ]),
                                SynapseList("first", "first", [
                                    Synapse("v", W_I, T_SYN)
                                ]),
                                SynapseList("input", "last", [
                                    Synapse("v", 0.5 * W_E, T_SYN)
                                ]),
                                SynapseList("first", "acc", [
                                    Synapse("ge", W_ACC, T_SYN)
                                ]),
                                SynapseList("acc", "acctwo", [
                                    Synapse("ge", -W_ACC, T_SYN)
                                ]),
                                SynapseList("last", "acctwo", [
                                    Synapse("ge", W_ACC, T_SYN)
                                ]),
                                SynapseList("acc", "ready", [
                                    Synapse("v", W_E, T_SYN)
                                ]),
                                SynapseList("recall", "acctwo", [
                                    Synapse("ge", W_ACC, T_SYN)
                                ]),
                                SynapseList("recall", "output", [
                                    Synapse("v", W_E, T_SYN)
                                ]),
                                SynapseList("acctwo", "output", [
                                    Synapse("v", W_E, T_SYN)
                                ])
                            ])

full_subtractor = Network(1, {"output+", "output-"},
                          ["inputone", "inputtwo", "syncone",
                           "synctwo", "inbone", "inbtwo", "output+",
                           "output-", "zero"],
                          [
                              SynapseList("inputone", "syncone", [
                                  Synapse("v", 0.5 * W_E, T_SYN)
                              ]),
                              SynapseList("inputtwo", "synctwo", [
                                  Synapse("v", 0.5 * W_E, T_SYN)
                              ]),
                              SynapseList("syncone", "output+", [
                                  Synapse("v", W_E,
                                          T_MIN + 3 * T_SYN + 2 * T_NEU)
                              ]),
                              SynapseList("syncone", "inbone", [
                                  Synapse("v", W_E, T_SYN)
                              ]),
                              SynapseList("syncone", "output-", [
                                  Synapse("v", W_E, 3 * T_SYN + 2 * T_NEU)
                              ]),
                              SynapseList("syncone", "inbtwo", [
                                  Synapse("v", W_I, T_SYN)
                              ]),
                              SynapseList("synctwo", "inbone", [
                                  Synapse("v", W_I, T_SYN)
                              ]),
                              SynapseList("synctwo", "output+", [
                                  Synapse("v", W_E, 3 * T_SYN + 2 * T_NEU)
                              ]),
                              SynapseList("synctwo", "inbtwo", [
                                  Synapse("v", W_E, T_SYN)
                              ]),
                              SynapseList("synctwo", "output-", [
                                  Synapse("v", W_E,
                                          T_MIN + 3 * T_SYN + 2 * T_NEU)
                              ]),
                              SynapseList("inbone", "output+", [
                                  Synapse("v", 2 * W_I, T_SYN)
                              ]),
                              SynapseList("inbtwo", "output-", [
                                  Synapse("v", 2 * W_I, T_SYN)
                              ]),
                              SynapseList("output+", "inbtwo", [
                                  Synapse("v", 0.5 * W_E, T_SYN)
                              ]),
                              SynapseList("output-", "inbone", [
                                  Synapse("v", 0.5 * W_E, T_SYN)
                              ]),
                              SynapseList("zero", "zero", [
                                  Synapse("v", W_E, T_NEU)
                              ]),
                              SynapseList("synctwo", "zero", [
                                  Synapse("v", 0.5 * W_E, T_NEU),
                                  Synapse("v", 0.5 * W_I, 2 * T_NEU)
                              ]),
                              SynapseList("syncone", "zero", [
                                  Synapse("v", 0.5 * W_E, T_NEU),
                                  Synapse("v", 0.5 * W_I, 2 * T_NEU)
                              ]),
                              SynapseList("zero", "inbtwo", [
                                  Synapse("v", W_I, T_NEU),
                              ]),
                              SynapseList("zero", "output-", [
                                  Synapse("v", 2 * W_I, T_NEU),
                              ])
                          ])
