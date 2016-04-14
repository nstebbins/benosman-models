#!/usr/bin/python

import socket

def parse_message(input):
    translation_table = dict.fromkeys(map(ord, '\n'), None)
    return(input.decode('ascii').translate(translation_table))

host = "0.0.0.0"
port = 8000

tcpsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
tcpsock.connect((host, port))
tcpsock.send("inverting_memory;input 2000 input 2900 recall 5000".encode())
output_v = parse_message(tcpsock.recv(4096))
print(len(output_v))
tcpsock.close()
