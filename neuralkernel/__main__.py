from .graph import plot_neurons
from .spikekernel import simulate_neurons


def main() -> None:
    # data = {'function': 'maximum', 'initial_values': {'input': [2000,
    # 2400], 'inputtwo': [2000, 2900]}}
    # data = {'function': 'inverting_memory', 'initial_values': {'input': [
    # 2000, 2900], 'recall': [5000]}}
    # data = {'function': 'non_inverting_memory', 'initial_values': {
    # 'input': [2000, 2200], 'recall': [5000]}}
    # data = {'function': 'full_subtractor', 'initial_values': {'inputone':
    # [2000, 2900], 'inputtwo': [2000, 2400]}}
    output_neurons = simulate_neurons("logarithm",
                                      {'input': [200, 270]})
    plot_neurons(output_neurons)


if __name__ == '__main__':
    main()
