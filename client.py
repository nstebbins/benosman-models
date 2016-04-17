import numpy as np
import socket, pickle
import spikekernel, tcp

def main():

    host = "0.0.0.0"
    port = 8000

    # create socket
    tcpsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcpsock.connect((host, port))

    messages = [
        "synchronizer;input0 1000 input0 1600 input1 2000 input1 2700"
    ]

    '''
    messages = [
        "inverting_memory;input 2000 input 2900 recall 5000",
        "logarithm;input 2000 input 2700",
        "maximum;input 2000 input 2700 input2 2000 input2 3500",
        "non_inverting_memory;input 2000 input 2200 recall 5000"]
    '''

    for message in messages:
        # send message
        tcp.send_msg(tcpsock, message.encode('ascii', 'ignore'))

        # receive message
        neurons = pickle.loads(tcp.recv_msg(tcpsock))
        spikekernel.plot_v(neurons)

    tcpsock.close()

main()
