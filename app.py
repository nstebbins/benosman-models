import neural.spikekernel as spikekernel
import numpy as np


def main():

    # messages = [
    #     "integrator;start 100 input- 3000 input- 4000 input- 5200 input- 5300 init 100 init 600",
    #     "logarithm;input 2000 input 2700",
    #     "maximum;input 2000 input 2400 inputtwo 2000 inputtwo 2900",
    #     "inverting_memory;input 2000 input 2900 recall 5000",
    #     "non_inverting_memory;input 2000 input 2200 recall 5000",
    #     "full_subtractor;inputone 2000 inputone 2900 inputtwo 2000 inputtwo 2400",
    # ]
    outputs, neurons = spikekernel.simulate_neurons("logarithm", {'input': [2000, 2700]})
    spikekernel.plot_v(np.take(neurons, outputs))


main()
