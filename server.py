import socket, threading, pickle
import spikekernel, tcp
import numpy as np

class client_thread(threading.Thread):

    def __init__(self, ip, port, socket):
        threading.Thread.__init__(self)
        self.ip = ip
        self.port = port
        self.socket = socket
        print("[+] new thread started for " + ip + ":" + str(port))

    def parse_message(self, input): # binary to string to (string, dict)

        input_str = input.decode('ascii') # convert to string

        # e.g. INVERTING_MEMORY;INPUT 2000 INPUT 2900 RECALL 5000
        f_specs = input_str.split(";")
        f_name = f_specs[0]; f_inputs = f_specs[1].split(" ")
        data = {}

        for i in range(0, len(f_inputs), 2):
            if f_inputs[i] in data:
                (data[f_inputs[i]]).append(int(f_inputs[i+1]))
            else:
                data[f_inputs[i]] = [int(f_inputs[i+1])]

        return((f_name, data))

    def run(self):

        print("[connection from]: " + self.ip + ":" + str(self.port))
        data = "dummydata" # @debug: can also configure do-while

        while True:
            rcv = tcp.recv_msg(self.socket)
            if rcv:
                f_name, data = self.parse_message(rcv)
                outputs, neurons = spikekernel.simulate_neurons(f_name, data)
                tcp.send_msg(self.socket,
                    pickle.dumps(np.take(neurons, outputs), protocol = 0))

            else:
                break

        print("client disconnected...")

def main():
    host = "0.0.0.0"
    port = 8000

    tcpsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # note: specify the option name as well as level below;
    # below option is effectively workaround for "ADDR ALREADY IN USE"
    tcpsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    tcpsock.bind((host,port))
    threads = []

    tcpsock.listen(4) # arg: max # of queued connections allowed

    while True: # server keeps running
        print("\nlistening for incoming connections...")
        (clientsock, (ip, port)) = tcpsock.accept()
        newthread = client_thread(ip, port, clientsock)
        newthread.start() # start thread to handle client
        threads.append(newthread)

    for t in threads:
        t.join()

main()
