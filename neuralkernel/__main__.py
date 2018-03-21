from .spikekernel import simulate_neurons, plot_output_neurons


def main() -> None:
    data = {'function': 'logarithm', 'initial_values': {'input': [2000, 2700]}}
    # data = {'function': 'maximum', 'initial_values': {'input': [2000, 2400], 'inputtwo': [2000, 2900]}}
    # data = {'function': 'inverting_memory', 'initial_values': {'input': [2000, 2900], 'recall': [5000]}}
    # data = {'function': 'non_inverting_memory', 'initial_values': {'input': [2000, 2200], 'recall': [5000]}}
    # data = {'function': 'full_subtractor', 'initial_values': {'inputone': [2000, 2900], 'inputtwo': [2000, 2400]}}
    outputs, neurons = simulate_neurons(data)
    plot_output_neurons(neurons, outputs)


if __name__ == '__main__':
    main()
