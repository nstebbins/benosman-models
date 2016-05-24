
import numpy as np
import socket, pickle

import tcp.tcp as tcp
import neural.spikekernel as spikekernel

def main():

    host = "0.0.0.0"
    port = 8000

    # create socket
    tcpsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcpsock.connect((host, port))

    messages = [
        "linear_combination;inputone+ 1000 inputone+ 1200 inputtwo- 1600 inputtwo- 2200"
    ]

    '''
    messages = [
        "logarithm;input 2000 input 2700",
        "maximum;input 2000 input 2400 inputtwo 2000 inputtwo 2900",
        "inverting_memory;input 2000 input 2900 recall 5000",
        "non_inverting_memory;input 2000 input 2200 recall 5000",
        "synchronizer;inputone 1000 inputone 1600 inputtwo 2000 inputtwo 2700",
        "full_subtractor;inputone 2000 inputone 2900 inputtwo 2000 inputtwo 2400",
        "linear_combination;input0+ 1000 input0+ 1200 input1- 1600 input1- 2200"
    ]
    '''

    for message in messages:
        # send message
        tcp.send_msg(tcpsock, message.encode('ascii', 'ignore'))

        # receive message
        neurons = pickle.loads(tcp.recv_msg(tcpsock))
        spikekernel.plot_v(neurons)

    tcpsock.close()

main()
