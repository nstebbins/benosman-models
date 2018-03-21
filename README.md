# Neuralkernel
[![Build Status](https://travis-ci.com/nstebbins/benosman-models.svg?token=wq8kpkt8TaRN17x6BNtj&branch=master)](https://travis-ci.com/nstebbins/benosman-models)
[![PyPI](https://img.shields.io/pypi/v/neuralkernel.svg)](https://pypi.python.org/pypi/neuralkernel)
[![PyPI - License](https://img.shields.io/pypi/l/neuralkernel.svg)](https://pypi.python.org/pypi/neuralkernel)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/neuralkernel.svg)](https://pypi.python.org/pypi/neuralkernel)

This project uses networks of neuron-like computational units to build a framework of computation. Specifically, it implements characteristics traditionally found in neural networks including synaptic diversity, temporal delays, and voltage spikes. It builds on the ideas proposed in the paper [STICK: Spike Time Interval Computational Kernel, A Framework for General Purpose Computation](https://arxiv.org/abs/1507.06222).

## Getting Started

To test various networks, you just need to run the following.

```bash
python -m neuralkernel
```

The networks currently implemented are:

* Inverting Memory
* Logarithm
* Maximum
* Non-Inverting Memory
* Full Subtractor

For more information on each of these networks, please check out the `docs` folder. 
