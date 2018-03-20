import neural.spikekernel as spikekernel


def main():
    data = {'function': 'logarithm', 'initial_values': {'input': [2000, 2700]}}
    # data = {'function': 'maximum', 'initial_values': {'input': [2000, 2400], 'inputtwo': [2000, 2900]}}
    # data = {'function': 'inverting_memory', 'initial_values': {'input': [2000, 2900], 'recall': [5000]}}
    # data = {'function': 'non_inverting_memory', 'initial_values': {'input': [2000, 2200], 'recall': [5000]}}
    # data = {'function': 'full_subtractor', 'initial_values': {'inputone': [2000, 2900], 'inputtwo': [2000, 2400]}}
    outputs, neurons = spikekernel.simulate_neurons(data)
    spikekernel.plot_output_neurons(neurons, outputs)


main()
