## benosman-models

This project uses modeling of neural networks as a general-purpose computational framework. Specifically, this project is interested in replicating operations on continuous-time signals. It builds off of the ideas and networks outlined by Xavier Lagorce and Ryad Benosman.

### Requirements

* Python 3

### Usage

To test various networks, you will need to run both a client (with operation requests) and a server. The server is multithreaded and allows a maximum of four queued connections at a time. 

```bash
python server.py
```

```bash
python client.py
```

The networks currently implemented are: 

* Inverting Memory
* Logarithm
* Maximum
* Non-Inverting Memory
* Synchronizer

For more information on each of these networks, please check out the `docs` folder. 
