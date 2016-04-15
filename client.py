import numpy as np
import socket, pickle
import spikekernel, tcp

def main():

    host = "0.0.0.0"
    port = 8000

    # create socket
    tcpsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcpsock.connect((host, port))

    # send parameters
    message = "inverting_memory;input 2000 input 2900 recall 5000"
    tcp.send_msg(tcpsock, message.encode('ascii', 'ignore'))

    # receive output voltage (as a numpy array)
    output_neuron = pickle.loads(tcp.recv_msg(tcpsock))
    spikekernel.plot_v(np.asarray([output_neuron]))

    tcpsock.close()

main()
